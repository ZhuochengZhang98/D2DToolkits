import os
import bz2
import logging
import pickle
from functools import lru_cache
from itertools import zip_longest
from typing import Iterable, Union, Dict, List, Any
from tempfile import TemporaryDirectory, TemporaryFile

import lmdb
import numpy as np
from fairseq.data import FairseqDataset


class LMDBDataset(FairseqDataset):
    def __init__(self, data_path: str):
        self.load_database(path=data_path)
        self._logger = logging.getLogger(self.__class__.__name__)
        return

    def load_database(self, path: str):
        # set basic args
        self._path = path
        # load mmap
        meta_path = f"{path}.pkl"
        data_path = f"{path}.lmdb"
        # load metadata
        self._meta = pickle.loads(open(meta_path, "rb").read())
        self._keys = self._meta["keys"]
        self._compress = self._meta["compress"]
        self._length = self._meta["length"]
        # memory map
        self._data_base = lmdb.Environment(data_path, readonly=True, lock=False)
        return

    def __getstate__(self):
        return {"path": self._path}

    def __setstate__(self, state):
        self.load_database(**state)

    @lru_cache(maxsize=8)
    def __getitem__(self, key: Union[bytes, int, np.signedinteger]):
        if int(key) >= self._length:
            raise StopIteration
        if isinstance(key, int) or isinstance(key, np.signedinteger):
            key = str(key).encode("ascii")
        with self._data_base.begin() as txn:
            data = txn.get(key, None)
            if data is None:
                raise KeyError(f"Invalid key:{key}")
            if self._compress:
                data = pickle.loads(bz2.decompress(data))
            else:
                data = pickle.loads(data)
        return data

    def get_items(self, items: Iterable[Union[bytes, int]]) -> List[Dict]:
        datas = []
        with self._data_base.begin() as txn:
            for item in items:
                data = txn.get(item, None)
                assert data is not None, f"Key error: {item.decode()}"
                if self._compress:
                    data = pickle.loads(bz2.decompress(data))
                else:
                    data = pickle.loads(data)
                datas.append(data)
        return datas

    def __del__(self):
        self._data_base.close()

    def __len__(self):
        return self._length

    @property
    def supports_prefetch(self):
        return False

    @property
    def keys(self):
        return self._keys


def merge_lmdb(
    data_paths: List[str],
    new_data_path: str,
):
    meta_paths = [f"{i}.pkl" for i in data_paths]
    data_paths = [f"{i}.lmdb" for i in data_paths]
    for meta_path, data_path in zip(meta_paths, data_paths):
        ...
    ...


def serialize_item(item: Any, compress: bool) -> bytes:
    """Special optimize for single key data

    Args:
        item (Any): item to be serialized
        compress (bool): compress the serialized data
    """
    if isinstance(item, np.ndarray):
        dumped = item.tobytes()
    else:
        dumped = pickle.dumps(item)
    if compress:
        dumped = bz2.compress(dumped)
    return dumped


def serialize(items: List[Any], keys: List[str], compress: bool) -> bytes:
    # if len(keys) == 1:
    #     return serialize_item(items[0], compress)
    dumped = {}
    for key, item in zip(keys, items):
        assert item is not None
        dumped[key] = item
    dumped = pickle.dumps(dumped)
    if compress:
        dumped = bz2.compress(dumped)
    return dumped


def dump(
    dumped_path: str,
    meta: Dict = {},
    dataset_len: int = None,
    logger=None,
    compress: bool = False,
    dry_run: bool = False,
    max_size: int = int(1e12),
    log_interval: int = 10000,
    **additional_info: Iterable,
) -> None:
    logger = logging.getLogger("dump") if logger is None else logger
    logger.info("Dumping LMDB data...")
    # prepare to dump
    if dry_run:
        data_path = TemporaryDirectory(prefix="lmdb_data")
        meta_file = TemporaryFile(mode="wb", prefix="lmdb_data")
        data_base = lmdb.Environment(data_path.name, map_size=max_size)
    else:
        data_path = f"{dumped_path}.lmdb"
        meta_path = f"{dumped_path}.pkl"
        if os.path.exists(data_path):
            raise FileExistsError(f"File exist: {data_path}")
        if os.path.exists(meta_path):
            raise FileExistsError(f"File exist: {meta_path}")
        meta_file = open(meta_path, "wb")
        data_base = lmdb.Environment(data_path, map_size=max_size)

    # dump the binary
    num = 0
    with data_base.begin(write=True) as txn:
        keys = list(additional_info.keys())
        for items in zip_longest(*additional_info.values(), fillvalue=None):
            if num % log_interval == 0:
                logger.info(f"Finish writing {num} items.")
            dumped = serialize(items, keys, compress)
            txn.put(str(num).encode("ascii"), dumped)
            num += 1
    data_base.close()

    # check length
    if dataset_len is not None:
        assert num == dataset_len
    # dump meta information
    meta.update({
        "length": num,
        "keys": keys,
        "compress": compress,
    })
    meta_file.write(pickle.dumps(meta))
    meta_file.close()
    logger.info(f"Finished binarize.")

    # clean work place
    if isinstance(data_path, TemporaryDirectory):
        data_path.cleanup()
    return
