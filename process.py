import argparse
import os
import sys
from typing import List

from torchrec.datasets.criteo import BinaryCriteoUtils


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Criteo tsv -> npy preprocessing script."
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    """
    This function preprocesses the raw Criteo tsvs into the format (npy binary)
    expected by InMemoryBinaryCriteoIterDataPipe.
    Args:
        argv (List[str]): Command line args.
    Returns:
        None.
    """

    for i in range(1):
        in_file_path = os.path.join("train.txt")
        if not os.path.exists(in_file_path):
            continue
        dense_out_file_path = os.path.join("torchrec_dense.npy")
        sparse_out_file_path = os.path.join("torchrec_sparse.npy")
        labels_out_file_path = os.path.join("torchrec_labels.npy")
        print(
            f"Processing {in_file_path}.\nOutput will be saved to\n{dense_out_file_path}"
            f"\n{sparse_out_file_path}\n{labels_out_file_path}"
        )
        BinaryCriteoUtils.tsv_to_npys(
            in_file_path,
            dense_out_file_path,
            sparse_out_file_path,
            labels_out_file_path,
        )
        print(f"Done processing {in_file_path}.")


if __name__ == "__main__":
    main(sys.argv[1:])