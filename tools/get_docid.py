import logging
import numpy as np
from argparse import ArgumentParser, Namespace


logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def main(args: Namespace):
    # prepare files
    src = open(args.src_path, "r")
    tgt = open(args.tgt_path, "r")
    src_dump = open(args.src_dump, "w")
    tgt_dump = open(args.tgt_dump, "w")
    doc_ids = []

    # steam process
    doc_id = 0
    for n, (src_line, tgt_line) in enumerate(zip(src, tgt)):
        if n % args.log_interval == 0:
            logger.info(f"Processing lines: {n}")
        # reach doc end
        if src_line.strip() == args.doc_spliter:
            assert tgt_line.strip() == args.doc_spliter
            if (n != 0):
                doc_id += 1
            continue
        src_dump.write(src_line)
        tgt_dump.write(tgt_line)
        doc_ids.append(doc_id)   

    # post process
    docid_file = np.memmap(
        args.docid_dump,
        dtype=int,
        mode="w+",
        shape=(len(doc_ids),),
    )
    docid_file[:] = np.array(doc_ids)[:]
    logger.info(f"Processed lines: {n}")
    logger.info(f"Processed documents: {doc_id}")
    src_dump.close()
    tgt_dump.close()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src-path",
        type=str,
        required=True,
        help="path to raw source file.",
    )
    parser.add_argument(
        "--tgt-path",
        type=str,
        required=True,
        help="path to raw target file.",
    )
    parser.add_argument(
        "--src-dump",
        type=str,
        required=True,
        help="path to dump source sentences.",
    )
    parser.add_argument(
        "--tgt-dump",
        type=str,
        required=True,
        help="path to dump target sentences.",
    )
    parser.add_argument(
        "--docid-dump",
        type=str,
        required=True,
        help="path to dump doc_ids.",
    )
    parser.add_argument(
        "--doc-spliter",
        type=str,
        default="<d>",
        help="spliter to split the source document",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100000,
        help="log every N lines",
    )
    args = parser.parse_args()
    main(args)
