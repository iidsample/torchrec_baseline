import argparse
import os
import logging
import time
import abc
import logging
import statistics
from dataclasses import dataclass, field
from typing import (
    Any,
    cast,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import torch
from torch.autograd.profiler import record_function
from torch.fx.node import Node
from torchrec.distributed.model_parallel import DistributedModelParallel, ShardedModule
from torchrec.distributed.types import Awaitable
from torchrec.modules.feature_processor import BaseGroupedFeatureProcessor
from torchrec.streamable import Multistreamable, Pipelineable

import torch
from torch import distributed as dist, nn
from torch.distributed.elastic.multiprocessing.errors import record
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec import EmbeddingBagCollection
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.datasets.criteo import InMemoryBinaryCriteoIterDataPipe
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper

from torch.autograd.profiler import record_function
from torch.fx.node import Node
from torchrec.distributed.model_parallel import DistributedModelParallel, ShardedModule
from torchrec.distributed.types import Awaitable
from torchrec.modules.feature_processor import BaseGroupedFeatureProcessor
from torchrec.streamable import Multistreamable, Pipelineable

from torchrec.datasets.utils import Batch
from torchrec.modules.crossnet import LowRankCrossNet
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torch.utils.data import DataLoader

from datetime import datetime
from s3_utils import uploadFile

import abc
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    cast,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from torch.autograd.profiler import record_function
from torch.fx.node import Node
from torchrec.distributed.model_parallel import DistributedModelParallel, ShardedModule
from torchrec.distributed.types import Awaitable
from torchrec.modules.feature_processor import BaseGroupedFeatureProcessor
from torchrec.streamable import Multistreamable, Pipelineable

In = TypeVar("In", bound=Pipelineable)
Out = TypeVar("Out")


class TrainPipeline(abc.ABC, Generic[In, Out]):
    @abc.abstractmethod
    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        pass


def _to_device(batch: In, device: torch.device, non_blocking: bool) -> In:
    assert isinstance(
        batch, (torch.Tensor, Pipelineable)
    ), f"{type(batch)} must implement Pipelineable interface"
    return cast(In, batch.to(device=device, non_blocking=non_blocking))


def _wait_for_batch(batch: In, stream: Optional[torch.cuda.streams.Stream]) -> None:
    if stream is None:
        return
    torch.cuda.current_stream().wait_stream(stream)
    # As mentioned in https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html,
    # PyTorch uses the "caching allocator" for memroy allocation for tensors. When a tensor is
    # freed, its memory is likely to be reused by newly constructed tenosrs.  By default,
    # this allocator traces whether a tensor is still in use by only the CUDA stream where it
    # was created.   When a tensor is used by additional CUDA streams, we need to call record_stream
    # to tell the allocator about all these streams.  Otherwise, the allocator might free the
    # underlying memory of the tensor once it is no longer used by the creator stream.  This is
    # a notable programming trick when we write programs using multi CUDA streams.
    cur_stream = torch.cuda.current_stream()
    assert isinstance(
        batch, (torch.Tensor, Multistreamable)
    ), f"{type(batch)} must implement Multistreamable interface"
    batch.record_stream(cur_stream)


class TrainPipelineBase(TrainPipeline[In, Out]):
    """
    This class runs training iterations using a pipeline of two stages, each as a CUDA
    stream, namely, the current (default) stream and `self._memcpy_stream`. For each
    iteration, `self._memcpy_stream` moves the input from host (CPU) memory to GPU
    memory, and the default stream runs forward, backward, and optimization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if device.type == "cuda" else None
        )
        self._cur_batch: Optional[In] = None
        self._connected = False

    def _connect(self, dataloader_iter: Iterator[In]) -> None:
        cur_batch = next(dataloader_iter)
        self._cur_batch = cur_batch
        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)
        self._connected = True

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        if not self._connected:
            self._connect(dataloader_iter)

        # Fetch next batch
        with record_function("## next_batch ##"):
            next_batch = next(dataloader_iter)
        cur_batch = self._cur_batch
        assert cur_batch is not None

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cur_batch, self._memcpy_stream)

        with record_function("## forward ##"):
            (losses, output) = self._model(cur_batch)

        if self._model.training:
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

        # Copy the next batch to GPU
        self._cur_batch = cur_batch = next_batch
        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)

        # Update
        if self._model.training:
            with record_function("## optimizer ##"):
                self._optimizer.step()

        return output


class Tracer(torch.fx.Tracer):
    # Disable proxying buffers during tracing. Ideally, proxying buffers would
    # be disabled, but some models are currently mutating buffer values, which
    # causes errors during tracing. If those models can be rewritten to not do
    # that, we can likely remove this line
    proxy_buffer_attributes = False

    def __init__(self, leaf_modules: Optional[List[str]] = None) -> None:
        super().__init__()
        self._leaf_modules: List[str] = leaf_modules if leaf_modules is not None else []

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, ShardedModule) or module_qualified_name in self._leaf_modules:
            return True
        return super().is_leaf_module(m, module_qualified_name)


@dataclass
class TrainPipelineContext:
    # pyre-ignore [4]
    input_dist_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    module_contexts: Dict[str, Multistreamable] = field(default_factory=dict)
    # pyre-ignore [4]
    feature_processor_forwards: List[Any] = field(default_factory=list)


@dataclass
class ArgInfo:
    # attributes of input batch, e.g. batch.attr1.attr2 call
    # will produce ["attr1", "attr2"]
    input_attrs: List[str]
    # batch[attr1].attr2 will produce [True, False]
    is_getitems: List[bool]
    # name for kwarg of pipelined forward() call or None
    # for a positional arg
    name: Optional[str]


class PipelinedForward:
    def __init__(
        self,
        name: str,
        args: List[ArgInfo],
        module: ShardedModule,
        context: TrainPipelineContext,
        dist_stream: Optional[torch.cuda.streams.Stream],
    ) -> None:
        self._name = name
        self._args = args
        self._module = module
        self._context = context
        self._dist_stream = dist_stream

    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        assert self._name in self._context.input_dist_requests
        request = self._context.input_dist_requests[self._name]
        assert isinstance(request, Awaitable)
        with record_function("## wait_sparse_data_dist ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with torch.cuda.stream(self._dist_stream):
                data = request.wait()

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        if self._dist_stream is not None:
            torch.cuda.current_stream().wait_stream(self._dist_stream)
            cur_stream = torch.cuda.current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
            data.record_stream(cur_stream)

            ctx = self._context.module_contexts[self._name]
            ctx.record_stream(cur_stream)

        if len(self._context.feature_processor_forwards) > 0:
            with record_function("## feature_processor ##"):
                for sparse_feature in data:
                    if sparse_feature.id_score_list_features is not None:
                        for fp_forward in self._context.feature_processor_forwards:
                            sparse_feature.id_score_list_features = fp_forward(
                                sparse_feature.id_score_list_features
                            )

        return self._module.compute_and_output_dist(
            self._context.module_contexts[self._name], data
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> List[ArgInfo]:
        return self._args


def _start_data_dist(
    pipelined_modules: List[ShardedModule],
    batch: In,
    context: TrainPipelineContext,
) -> None:
    context.input_dist_requests.clear()
    context.module_contexts.clear()
    for module in pipelined_modules:
        forward = module.forward
        assert isinstance(forward, PipelinedForward)

        # Retrieve argument for the input_dist of EBC
        # is_getitem True means this argument could be retrieved by a list
        # False means this argument is getting while getattr
        # and this info was done in the _rewrite_model by tracing the
        # entire model to get the arg_info_list
        args = []
        kwargs = {}
        for arg_info in forward.args:
            if arg_info.input_attrs:
                arg = batch
                for attr, is_getitem in zip(arg_info.input_attrs, arg_info.is_getitems):
                    if is_getitem:
                        arg = arg[attr]
                    else:
                        arg = getattr(arg, attr)
                if arg_info.name:
                    kwargs[arg_info.name] = arg
                else:
                    args.append(arg)
            else:
                args.append(None)
        # Start input distribution.
        module_ctx = module.create_context()
        context.module_contexts[forward.name] = module_ctx
        context.input_dist_requests[forward.name] = module.input_dist(
            module_ctx, *args, **kwargs
        )


def _get_node_args_helper(
    # pyre-ignore
    arguments,
    num_found: int,
    feature_processor_arguments: Optional[List[Node]] = None,
) -> Tuple[List[ArgInfo], int]:
    """
    Goes through the args/kwargs of a node and arranges them into a list of `ArgInfo`s.
    It also counts the number of (args + kwargs) found.
    """

    arg_info_list = [ArgInfo([], [], None) for _ in range(len(arguments))]
    for arg, arg_info in zip(arguments, arg_info_list):
        if arg is None:
            num_found += 1
            continue
        while True:
            if not isinstance(arg, torch.fx.Node):
                break
            child_node = arg

            if child_node.op == "placeholder":
                num_found += 1
                break
            # skip this fp node
            elif (
                feature_processor_arguments is not None
                and child_node in feature_processor_arguments
            ):
                arg = child_node.args[0]
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "builtins"
                # pyre-ignore[16]
                and child_node.target.__name__ == "getattr"
            ):
                arg_info.input_attrs.insert(0, child_node.args[1])
                arg_info.is_getitems.insert(0, False)
                arg = child_node.args[0]
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "_operator"
                # pyre-ignore[16]
                and child_node.target.__name__ == "getitem"
            ):
                arg_info.input_attrs.insert(0, child_node.args[1])
                arg_info.is_getitems.insert(0, True)
                arg = child_node.args[0]
            else:
                break
    return arg_info_list, num_found


def _get_node_args(
    node: Node, feature_processor_nodes: Optional[List[Node]] = None
) -> Tuple[List[ArgInfo], int]:
    num_found = 0
    pos_arg_info_list, num_found = _get_node_args_helper(
        node.args, num_found, feature_processor_nodes
    )
    kwargs_arg_info_list, num_found = _get_node_args_helper(
        node.kwargs.values(), num_found
    )

    # Replace with proper names for kwargs
    for name, arg_info_list in zip(node.kwargs, kwargs_arg_info_list):
        arg_info_list.name = name

    arg_info_list = pos_arg_info_list + kwargs_arg_info_list
    return arg_info_list, num_found


def _get_unsharded_module_names_helper(
    model: torch.nn.Module,
    path: str,
    unsharded_module_names: Set[str],
) -> bool:
    sharded_children = set()
    for name, child in model.named_children():
        curr_path = path + name
        if isinstance(child, ShardedModule):
            sharded_children.add(name)
        else:
            child_sharded = _get_unsharded_module_names_helper(
                child,
                curr_path + ".",
                unsharded_module_names,
            )
            if child_sharded:
                sharded_children.add(name)

    if len(sharded_children) > 0:
        for name, _ in model.named_children():
            if name not in sharded_children:
                unsharded_module_names.add(path + name)

    return len(sharded_children) > 0


def _get_unsharded_module_names(model: torch.nn.Module) -> List[str]:
    """
    Returns a list of top level modules do not contain any sharded sub modules.
    """

    unsharded_module_names: Set[str] = set()
    _get_unsharded_module_names_helper(
        model,
        "",
        unsharded_module_names,
    )
    return list(unsharded_module_names)


def _rewrite_model(  # noqa C901
    model: torch.nn.Module,
    context: TrainPipelineContext,
    dist_stream: Optional[torch.cuda.streams.Stream],
) -> List[ShardedModule]:

    # Get underlying nn.Module
    if isinstance(model, DistributedModelParallel):
        model = model.module

    # Collect a list of sharded modules.
    sharded_modules = {}
    fp_modules = {}
    for name, m in model.named_modules():
        if isinstance(m, ShardedModule):
            sharded_modules[name] = m
        if isinstance(m, BaseGroupedFeatureProcessor):
            fp_modules[name] = m

    # Trace a model.
    tracer = Tracer(leaf_modules=_get_unsharded_module_names(model))
    graph = tracer.trace(model)

    feature_processor_nodes = []
    # find the fp node
    for node in graph.nodes:
        if node.op == "call_module" and node.target in fp_modules:
            feature_processor_nodes.append(node)
    # Select sharded modules, which are top-level in the forward call graph,
    # i.e. which don't have input transformations, i.e.
    # rely only on 'builtins.getattr'.
    ret = []
    for node in graph.nodes:
        if node.op == "call_module" and node.target in sharded_modules:
            total_num_args = len(node.args) + len(node.kwargs)
            if total_num_args == 0:
                continue
            arg_info_list, num_found = _get_node_args(node, feature_processor_nodes)
            if num_found == total_num_args:
                child = sharded_modules[node.target]
                child.forward = PipelinedForward(
                    node.target,
                    arg_info_list,
                    child,
                    context,
                    dist_stream,
                )
                ret.append(child)
    return ret


def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for Python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class SparseArch(nn.Module):
    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection
        assert (
            self.embedding_bag_collection.embedding_bag_configs
        ), "Embedding bag collection cannot be empty!"
        self.D: int = self.embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        self._sparse_feature_names: List[str] = [
            name
            for conf in embedding_bag_collection.embedding_bag_configs()
            for name in conf.feature_names
        ]

        self.F: int = len(self._sparse_feature_names)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        sparse_features: KeyedTensor = self.embedding_bag_collection(features)

        B: int = features.stride()

        sparse: Dict[str, torch.Tensor] = sparse_features.to_dict()
        sparse_values: List[torch.Tensor] = []
        for name in self.sparse_feature_names:
            sparse_values.append(sparse[name])

        return torch.cat(sparse_values, dim=1).reshape(B, self.F, self.D)

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class DenseArch(nn.Module):
    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class InteractionArch(nn.Module):
    def __init__(self, num_sparse_features: int, emb_dim) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.emb_dim = emb_dim

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        flat_tensor = sparse_features.reshape((-1, self.F * self.emb_dim))
        
        comb = torch.cat([flat_tensor, dense_features], dim=1)

        return comb


class OverArch(nn.Module):
    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(in_features, layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class DLRM(nn.Module):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dense_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        assert (
            len(embedding_bag_collection.embedding_bag_configs()) > 0
        ), "At least one embedding bag is required"
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs())):
            conf_prev = embedding_bag_collection.embedding_bag_configs()[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs()[i]
            assert (
                conf_prev.embedding_dim == conf.embedding_dim
            ), "All EmbeddingBagConfigs must have the same dimension"
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim

        self.sparse_arch: SparseArch = SparseArch(embedding_bag_collection)
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=dense_device,
        )

        self.inter_arch = InteractionArch(
            num_sparse_features=num_sparse_features,
            emb_dim=embedding_dim
        )

        over_in_features: int = embedding_dim * num_sparse_features + dense_arch_layer_sizes[-1]
        

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )
        
    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
        dense_start, 
        dense_stop, 
        sparse_start, 
        sparse_stop, 
        inter_start, 
        inter_stop, 
        over_start, 
        over_stop
    ) -> torch.Tensor:
        dense_start.record()
        embedded_dense = self.dense_arch(dense_features)
        dense_stop.record()
        sparse_start.record()
        embedded_sparse = self.sparse_arch(sparse_features)
        sparse_stop.record()
        inter_start.record()
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        inter_stop.record()
        over_start.record()
        logits = self.over_arch(concatenated_dense)
        over_stop.record()
        return logits


class DLRMTrain(nn.Module):
    def __init__(
        self,
        dlrm_module: DLRM,
    ) -> None:
        super().__init__()
        self.model = dlrm_module
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(
        self, batch: Batch, dense_start, dense_stop, sparse_start, sparse_stop, inter_start, inter_stop, over_start, over_stop
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        logits = self.model(batch.dense_features, batch.sparse_features, dense_start, dense_stop, sparse_start, sparse_stop, inter_start, inter_stop, over_start, over_stop)
        logits = logits.squeeze(-1)
        loss = self.loss_fn(logits, batch.labels.float())

        return loss, (loss.detach(), logits.detach(), batch.labels.detach())


class TrainPipelineSparseDist(TrainPipeline[In, Out]):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._device = device
        # use two data streams to support two concurrent batches
        if device.type == "cuda":
            self._memcpy_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.Stream()
            self._data_dist_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.Stream()
        else:
            self._memcpy_stream: Optional[torch.cuda.streams.Stream] = None
            self._data_dist_stream: Optional[torch.cuda.streams.Stream] = None
        self._batch_i: Optional[In] = None
        self._batch_ip1: Optional[In] = None
        self._batch_ip2: Optional[In] = None
        self._connected = False
        self._context = TrainPipelineContext()
        self._pipelined_modules: List[ShardedModule] = []

    def _replace_fp_forward(self, model: torch.nn.Module) -> None:
        for _, m in model.named_modules():
            if isinstance(m, BaseGroupedFeatureProcessor):
                self._context.feature_processor_forwards.append(m.forward)
                # pyre-ignore[8]: Incompatible attribute type
                m.forward = lambda x: x

    def _connect(self, dataloader_iter: Iterator[In]) -> None:
        self._replace_fp_forward(cast(torch.nn.Module, self._model.module))
        # batch 1
        with torch.cuda.stream(self._memcpy_stream):
            batch_i = next(dataloader_iter)
            self._batch_i = batch_i = _to_device(
                batch_i, self._device, non_blocking=True
            )
            # Try to pipeline input data dist.
            self._pipelined_modules = _rewrite_model(
                self._model, self._context, self._data_dist_stream
            )

        with torch.cuda.stream(self._data_dist_stream):
            _wait_for_batch(batch_i, self._memcpy_stream)
            _start_data_dist(self._pipelined_modules, batch_i, self._context)

        # batch 2
        with torch.cuda.stream(self._memcpy_stream):
            batch_ip1 = next(dataloader_iter)
            self._batch_ip1 = batch_ip1 = _to_device(
                batch_ip1, self._device, non_blocking=True
            )
        self._connected = True

    def progress(self, 
                 dataloader_iter: Iterator[In], 
                 zero_grad_start, 
                 zero_grad_stop, 
                 copy_batch_to_gpu_start, 
                 copy_batch_to_gpu_stop, 
                 wait_for_batch_start, 
                 wait_for_batch_stop, 
                 forward_start, 
                 forward_stop, 
                 sparse_data_dist_start, 
                 sparse_data_dist_stop, 
                 backward_start, 
                 backward_stop, 
                 optimizer_start, 
                 optimizer_stop,
                 dense_start,
                 dense_stop,
                 sparse_start,
                 sparse_stop,
                 inter_start,
                 inter_stop,
                 over_start,
                 over_stop) -> Out:
        if not self._connected:
            self._connect(dataloader_iter)

        if self._model.training:
            zero_grad_start.record()
            self._optimizer.zero_grad()
            zero_grad_stop.record()
        copy_batch_to_gpu_start.record()
        with torch.cuda.stream(self._memcpy_stream):
            batch_ip2 = next(dataloader_iter)
            self._batch_ip2 = batch_ip2 = _to_device(
                batch_ip2, self._device, non_blocking=True
            )
        copy_batch_to_gpu_stop.record()
        
        batch_i = cast(In, self._batch_i)
        batch_ip1 = cast(In, self._batch_ip1)

        wait_for_batch_start.record()
        _wait_for_batch(batch_i, self._data_dist_stream)
        wait_for_batch_stop.record()

        forward_start.record()
        if self._data_dist_stream:
            event = torch.cuda.current_stream().record_event()
        (losses, output) = cast(Tuple[torch.Tensor, Out], self._model(batch_i, dense_start, dense_stop, sparse_start, sparse_stop, inter_start, inter_stop, over_start, over_stop))
        forward_stop.record()

        sparse_data_dist_start.record()
        # Data Distribution
        with torch.cuda.stream(self._data_dist_stream):
            _wait_for_batch(batch_ip1, self._memcpy_stream)
            # Ensure event in default stream has been called before
            # starting data dist
            if self._data_dist_stream:
                # pyre-ignore [61]: Local variable `event` is undefined, or not always defined
                self._data_dist_stream.wait_event(event)
            _start_data_dist(self._pipelined_modules, batch_ip1, self._context)
        sparse_data_dist_stop.record()

        if self._model.training:
            # Backward
            backward_start.record()
            torch.sum(losses, dim=0).backward()
            backward_stop.record()

            optimizer_start.record()
            # Update
            self._optimizer.step()
            optimizer_stop.record()

        self._batch_i = batch_ip1
        self._batch_ip1 = batch_ip2

        return output


def parse_args(parser):
    parser.add_argument("--iter", type=int, default=2000, help="iterations to run")
    parser.add_argument("--batch-size", type=int, default=16384, help="batch size to use for training")
    parser.add_argument("--world-size-trainers", type=int, default=8, help="For logging only")
    parser.add_argument("--s3", action="store_true", default=False)
    parser.add_argument("--logging-prefix", type=str, default="TorchRec")
    parser.add_argument("--dataset", type=str, default="../torchrec_dataset")
    return parser.parse_args()


@record
def main(args) -> None:
    batch_size = args.batch_size
    num_dense = 13
    num_sparse = 26
    ln_emb = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]

    emb_dim = 48
    
    configs = [
        EmbeddingBagConfig(
            name=f"table{id}",
            embedding_dim=emb_dim,
            num_embeddings=ln_emb[id],
            feature_names=[f"cat_{id}"],
        )
        for id in range(num_sparse)
    ]

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    backend = "gloo"
    # backend = "nccl"
    torch.cuda.set_device(device)
    
    now = datetime.now().strftime("%H:%M_%B_%d_%Y")
    filename=f"training_worker_{os.environ['RANK']}_{now}_{args.logging_prefix}.log"
    logging.basicConfig(
        filename=filename
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(args)
    
    dist.init_process_group(backend=backend)

    model = DLRMTrain(DLRM(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=configs, device=torch.device("meta")
        ),
        dense_in_features=num_dense,
        dense_arch_layer_sizes=[256, 256, 256],
        over_arch_layer_sizes=[1],
        dense_device=device,
    ))
    
    fused_params = {
        "learning_rate": 0.01,
        "optimizer": OptimType.EXACT_SGD,
    }
    sharders = [
        EmbeddingBagCollectionSharder(fused_params=fused_params),
    ]
    
    model = DistributedModelParallel(
        module=model,
        device=device,
        sharders=cast(List[ModuleSharder[nn.Module]], sharders),
    )
    dense_optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.SGD(params, lr=0.01),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    
    zero_grad_start = torch.cuda.Event(enable_timing=True)
    zero_grad_stop = torch.cuda.Event(enable_timing=True)
    copy_batch_to_gpu_start = torch.cuda.Event(enable_timing=True)
    copy_batch_to_gpu_stop = torch.cuda.Event(enable_timing=True)
    wait_for_batch_start = torch.cuda.Event(enable_timing=True)
    wait_for_batch_stop = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_stop = torch.cuda.Event(enable_timing=True)
    dense_start = torch.cuda.Event(enable_timing=True)
    dense_stop = torch.cuda.Event(enable_timing=True)
    sparse_start = torch.cuda.Event(enable_timing=True)
    sparse_stop = torch.cuda.Event(enable_timing=True)
    inter_start = torch.cuda.Event(enable_timing=True)
    inter_stop = torch.cuda.Event(enable_timing=True)
    over_start = torch.cuda.Event(enable_timing=True)
    over_stop = torch.cuda.Event(enable_timing=True)
    sparse_data_dist_start = torch.cuda.Event(enable_timing=True)
    sparse_data_dist_stop = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_stop = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_stop = torch.cuda.Event(enable_timing=True)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_stop = torch.cuda.Event(enable_timing=True)
    
    train_pipeline = TrainPipelineSparseDist(
        model,
        optimizer,
        device,
    )
    
    template = args.dataset
    datapipe = DataLoader(
        InMemoryBinaryCriteoIterDataPipe(
            stage="train",
            dense_paths=[os.path.join(template, "torchrec_dense.npy")],
            sparse_paths=[os.path.join(template, "torchrec_sparse.npy")],
            labels_paths=[os.path.join(template, "torchrec_labels.npy")],
            batch_size=batch_size,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            # mmap_mode=args.mmap_mode,
            hashes=ln_emb,
        ),
        batch_size=None,
        pin_memory=True,
        collate_fn=lambda x: x,
    )
    it = iter(datapipe)
    train_pipeline._model.train()
    stop_iter = 0
    iter_list = []
    start_time = time.time() * 1000
    while True and stop_iter < args.iter:
        try:
            print(f"On batch {stop_iter}")
            logger.info(f"On batch {stop_iter}")
            iter_start.record()
            train_pipeline.progress(it, 
                                    zero_grad_start, 
                                    zero_grad_stop, 
                                    copy_batch_to_gpu_start, 
                                    copy_batch_to_gpu_stop, 
                                    wait_for_batch_start, 
                                    wait_for_batch_stop, 
                                    forward_start, 
                                    forward_stop, 
                                    sparse_data_dist_start, 
                                    sparse_data_dist_stop, 
                                    backward_start, 
                                    backward_stop, 
                                    optimizer_start, 
                                    optimizer_stop,
                                    dense_start,
                                    dense_stop,
                                    sparse_start,
                                    sparse_stop,
                                    inter_start,
                                    inter_stop,
                                    over_start,
                                    over_stop)
            iter_stop.record()
            torch.cuda.synchronize()
            logger.info("zero_grad Time {}ms".format(zero_grad_start.elapsed_time(zero_grad_stop)))
            logger.info("copy_batch_to_gpu Time {}ms".format(copy_batch_to_gpu_start.elapsed_time(copy_batch_to_gpu_stop)))
            logger.info("wait_for_batch Time {}ms".format(wait_for_batch_start.elapsed_time(wait_for_batch_stop)))
            logger.info("dense Time {}ms".format(dense_start.elapsed_time(dense_stop)))
            logger.info("sparse Time {}ms".format(sparse_start.elapsed_time(sparse_stop)))
            logger.info("inter Time {}ms".format(inter_start.elapsed_time(inter_stop)))
            logger.info("over Time {}ms".format(over_start.elapsed_time(over_stop)))
            logger.info("forward Time {}ms".format(forward_start.elapsed_time(forward_stop)))
            logger.info("sparse_data_dist Time {}ms".format(sparse_data_dist_start.elapsed_time(sparse_data_dist_stop)))
            logger.info("backward Time {}ms".format(backward_start.elapsed_time(backward_stop)))
            logger.info("optimizer Time {}ms".format(optimizer_start.elapsed_time(optimizer_stop)))
            iter_list.append(iter_start.elapsed_time(iter_stop))
            stop_iter += 1
        except StopIteration:
            break
    end_time = time.time() * 1000

    print(f"For a total of {stop_iter - 1} iterations, torchrec takes on average {statistics.mean(iter_list[1:])}ms with batch size of {batch_size}")
    logger.info(f"For a total of {stop_iter - 1} iterations, torchrec takes on average {statistics.mean(iter_list[1:])}ms with batch size of {batch_size}")
    logger.info(f"For a total of {stop_iter} iterations, torchrec takes {end_time - start_time}ms with batch size of {batch_size}")
    if args.s3:
        s3_resource = uploadFile("recommendation-data-bagpipe")
        s3_resource.push_file(filename, f"{args.world_size_trainers}_trainers_torchrec/{filename}")


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Arguments for W&D"))
    main(args)