import logging
from argparse import ArgumentParser, Namespace, FileType
from subprocess import Popen, PIPE

import numpy as np


logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def main(args: Namespace):
    docids = np.memmap(filename=args.docid, dtype=int, mode="r")
    doc_num = len(set(docids))

    stdout, _ = Popen(
        ["wc", args.src],
        stdout=PIPE,
    ).communicate()
    src_lines, src_toks, src_bytes, _ = stdout.decode("utf-8").split()
    stdout, _ = Popen(
        ["wc", args.tgt],
        stdout=PIPE,
    ).communicate()
    tgt_lines, tgt_toks, tgt_bytes, _ = stdout.decode("utf-8").split()

    

    logger.info(f"Source: {doc_num} docs, {src_lines} lines, {src_toks} tokens")
    logger.info(f"Target: {doc_num} docs, {tgt_lines} lines, {tgt_toks} tokens")
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
        help="path to raw target file.",
    )
    parser.add_argument(
        "--docid",
        type=str,
        required=True,
        help="path to document id file.",
    )
    args = parser.parse_args()
    main(args)
