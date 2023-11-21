import logging
from argparse import ArgumentParser, Namespace, FileType
from typing import TextIO, List
from subprocess import Popen

import numpy as np
import xml.etree.ElementTree as ET


logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def parse_id(idfile: TextIO) -> List[int]:
    logger.info(f"Processing idfile: {idfile.name}")
    if idfile.name.endswith("ids"):
        ids = []
        id = 0
        docs = set()
        lines = [id_line.strip().split("\t")[:2] for id_line in idfile]
        for src_doc, tgt_doc in lines:
            if (src_doc, tgt_doc) not in docs:
                id += 1
                docs.add((src_doc, tgt_doc))
            ids.append(id)
    else:
        assert idfile.name.endswith("xml")
        ids = []
        ids_root = ET.parse(idfile).getroot()
        for n, doc in enumerate(ids_root):
            ids.extend([n] * len(doc))
    logger.info(f"Processed documents: {len(set(ids))}")
    logger.info(f"Processed lines: {len(ids)}")
    return ids


def main(args: Namespace):
    doc_ids = parse_id(args.idfile)

    # post process
    docid_file = np.memmap(
        args.docid_dump,
        dtype=int,
        mode="w+",
        shape=(len(doc_ids),),
    )
    docid_file[:] = np.array(doc_ids)[:]

    # move text if indicated
    if (args.src is not None) and (args.src_dump is not None):
        Popen(["cp", args.src, args.src_dump]).wait()
    if (args.tgt is not None) and (args.tgt_dump is not None):
        Popen(["cp", args.tgt, args.tgt_dump]).wait()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--idfile",
        type=FileType("r"),
        required=True,
        help="path to raw docid file(ended with ids/xml).",
    )
    parser.add_argument(
        "--docid-dump",
        type=str,
        required=True,
        help="path to dump doc_ids.",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="path to raw source file.(Optional)",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        default=None,
        help="path to raw target file.(Optioanl)",
    )
    parser.add_argument(
        "--src-dump",
        type=str,
        default=None,
        help="path to dump source sentences.(Optional)",
    )
    parser.add_argument(
        "--tgt-dump",
        type=str,
        default=None,
        help="path to dump target sentences.(Optioanl)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100000,
        help="log every N lines",
    )
    args = parser.parse_args()
    main(args)
