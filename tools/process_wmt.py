import logging
from argparse import ArgumentParser, Namespace, FileType
import xml.etree.ElementTree as ET

import numpy as np

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def main(args: Namespace):
    src = ET.parse(args.src)
    tgt = ET.parse(args.tgt)
    src_root = src.getroot()
    tgt_root = tgt.getroot()

    # steam process
    doc_ids = []
    lines = 0
    for n, (src_doc, tgt_doc) in enumerate(zip(src_root, tgt_root)):
        for src_p, tgt_p in zip(src_doc, tgt_doc):
            for src_seg, tgt_seg in zip(src_p, tgt_p):
                lines += 1
                args.src_dump.write(src_seg.text.strip() + "\n")
                args.tgt_dump.write(tgt_seg.text.strip() + "\n")
                doc_ids.append(n)

    # post process
    docid_file = np.memmap(
        args.docid_dump,
        dtype=int,
        mode="w+",
        shape=(len(doc_ids),),
    )
    docid_file[:] = np.array(doc_ids)[:]
    logger.info(f"Processed lines: {lines}")
    logger.info(f"Processed documents: {n}")
    args.src_dump.close()
    args.tgt_dump.close()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="path to raw source file.",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        required=True,
        help="path to raw source file.",
    )
    parser.add_argument(
        "--src-dump",
        type=FileType("w"),
        required=True,
        help="path to dump source sentences.",
    )
    parser.add_argument(
        "--tgt-dump",
        type=FileType("w"),
        required=True,
        help="path to dump target sentences.",
    )
    parser.add_argument(
        "--docid-dump",
        type=str,
        required=True,
        help="path to dump doc_ids.",
    )
    args = parser.parse_args()
    main(args)
