import bz2
import logging
import pickle
from copy import deepcopy
from functools import lru_cache
from itertools import zip_longest
from mmap import ACCESS_READ, mmap
from typing import Iterable

import torch
from fairseq.data.indexed_dataset import _warmup_mmap_file


class ExternalIndexedData(torch.utils.data.Dataset):
    def __init__(
        self,
        external_path: str,
        warmup: bool = False,
        compress: bool = False,
    ):
        self._bin_buffer = None
        self._bin_file = None
        self._indices = None
        self._path = None
        self._warmup = None
        self.load_mmap(
            path=external_path,
            warmup=warmup,
            compress=compress,
        )
        self._logger = logging.getLogger(self.__class__.__name__)
        return

    def load_mmap(
        self,
        path: str,
        warmup: bool = False,
        compress: bool = False,
    ):
        # set basic args
        self._path = path
        self._warmup = warmup
        self._compress = compress
        # load mmap
        idx_path = f"{path}.pkl.idx"
        ext_path = f"{path}.pkl.bin"
        self._meta = pickle.loads(open(idx_path, "rb").read())
        self._indices = self._meta["indices"]
        self._keys = self._meta["keys"]
        # warmup if needed
        if warmup:
            self._logger.info("Warming up file...")
            _warmup_mmap_file(ext_path)
        # memory map
        self._bin_file = open(ext_path, "r")
        self._bin_buffer = mmap(self._bin_file.fileno(), 0, access=ACCESS_READ)
        return

    def __getstate__(self):
        return {
            "path": self._path,
            "warmup": self._warmup,
            "compress": self._compress,
        }

    def __setstate__(self, state):
        self.load_mmap(**state)

    @lru_cache(maxsize=8)
    def __getitem__(self, id: int):
        ext_data = self._bin_buffer[self._indices[id] : self._indices[id + 1]]
        if self._compress:
            ext_data = pickle.loads(bz2.decompress(ext_data))
        else:
            ext_data = pickle.loads(ext_data)
        return deepcopy(ext_data)

    def __del__(self):
        self._bin_buffer.close()
        self._bin_file.close()
        del self._bin_buffer
        del self._bin_file
        del self._indices

    def __len__(self):
        return len(self._indices)

    @property
    def supports_prefetch(self):
        return False

    @property
    def index(self):
        return self._indices

    @property
    def keys(self):
        return self._keys


def dump_addition(
    dumped_path: str,
    dataset_len: int = None,
    logger=None,
    compress:bool = False,
    dry_run:bool = False,
    **additional_info: Iterable,
) -> None:
    logger.info("Dumping external data...")
    # prepare to dump
    if dry_run:
        byt_path = "/dev/null"
        idx_path = None
    else:
        byt_path = f"{dumped_path}.pkl.bin"
        idx_path = f"{dumped_path}.pkl.idx"
    total = 0
    num = 0
    indices = [0]

    # dump the binary
    keys = list(additional_info.keys())
    with open(byt_path, "wb") as f:
        for items in zip_longest(*additional_info.values(), fillvalue=None):
            if num % 10000 == 0:
                logger.info(f"Finish writing {num} items.")
            dumped = {}
            for n, item in enumerate(items):
                assert item is not None
                dumped[keys[n]] = item
            dumped = pickle.dumps(dumped)
            if compress:
                dumped = bz2.compress(dumped)
            f.write(dumped)
            total += len(dumped)
            indices.append(total)
            num += 1

    # check length
    if dataset_len is not None:
        assert num == dataset_len
    # dump meta information
    meta = {
        "indices": indices,
        "keys": keys,
    }
    if idx_path is not None:
        with open(idx_path, "wb") as f:
            f.write(pickle.dumps(meta))
    logger.info(f"Finished binarize.")
    return
