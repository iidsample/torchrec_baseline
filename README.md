# TorchRec Baseline

```
TP_SOCKET_IFNAME=enp94s0f0 NCCL_SOCKET_IFNAME=enp94s0f0 torchrun --nnodes 2 --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost --rdzv_id 54321 --role trainer trainer_main.py
TP_SOCKET_IFNAME=enp94s0f0 NCCL_SOCKET_IFNAME=enp94s0f0 torchrun --nnodes 2 --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint 10.10.1.1 --rdzv_id 54321 --role trainer trainer_main.py
```