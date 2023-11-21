import time
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
from lmdb_dataset import LMDBDataset


def main(args: Namespace):
    dataset = LMDBDataset(args.data_path)
    start_time = time.time()
    for item in tqdm(dataset):
        pass
    end_time = time.time()
    print(f"Test Latency: {(end_time - start_time) / len(dataset)}")
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "data_path",
        type=str,
        help="path to the lmdb dataset",
    )
    args = parser.parse_args()
    main(args)
