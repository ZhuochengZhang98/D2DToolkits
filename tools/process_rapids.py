import logging
from argparse import ArgumentParser, Namespace, FileType
import xml.etree.ElementTree as ET

import numpy as np

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def main(args: Namespace):
    tree = ET.parse(args.data)
    root = tree.getroot()
    # steam process
    doc_ids = []
    lines = 0
    for n, document in enumerate(root):
        for unit in document:
            for seg in unit:
                src_line, tgt_line = None, None
                for sent in seg:
                    if "source" in sent.tag:
                        assert src_line is None
                        src_line = sent.text.strip()
                    else:
                        assert "target" in sent.tag
                        assert tgt_line is None
                        tgt_line = sent.text.strip()
                lines += 1
                args.src_dump.write(src_line + "\n")
                args.tgt_dump.write(tgt_line + "\n")
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
        "--data",
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
