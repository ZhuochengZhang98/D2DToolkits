import logging
from argparse import ArgumentParser, Namespace, FileType

import numpy as np


logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def main(args: Namespace):
    doc_ids = []
    new_docid = 0
    n = 0

    for src_file, tgt_file, docid_file in zip(args.src, args.tgt, args.docid):
        docids = np.memmap(filename=docid_file, dtype=int, mode="r")
        last_docid = 0
        for src_line, tgt_line, docid in zip(src_file, tgt_file, docids):
            if n % args.log_interval == 0:
                logger.info(f"Processing lines: {len(doc_ids)}")
            if docid != last_docid:
                new_docid += 1
                last_docid = docid
            doc_ids.append(new_docid)
            args.src_dump.write(src_line)
            args.tgt_dump.write(tgt_line)
            n += 1
        new_docid += 1

    # post process
    docid_file = np.memmap(
        args.docid_dump,
        dtype=int,
        mode="w+",
        shape=(len(doc_ids),),
    )
    docid_file[:] = np.array(doc_ids)[:]
    logger.info(f"Total lines: {len(doc_ids)}")
    logger.info(f"Total documents: {doc_ids[-1]}")
    args.src_dump.close()
    args.tgt_dump.close()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src",
        type=FileType("r"),
        nargs="+",
        required=True,
        help="path to raw source file.",
    )
    parser.add_argument(
        "--tgt",
        type=FileType("r"),
        nargs="+",
        required=True,
        help="path to raw target file.",
    )
    parser.add_argument(
        "--docid",
        type=str,
        nargs="+",
        required=True,
        help="path to document id file.",
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
        help="path to dump document id.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100000,
        help="log interval",
    )
    args = parser.parse_args()
    main(args)
