import logging
from argparse import ArgumentParser, Namespace, FileType

import numpy as np

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def main(args: Namespace):
    # steam process
    doc_id = 0
    doc_ids = []
    for n, row in enumerate(args.data):
        if n % args.log_interval == 0:
            logger.info(f"Processing lines: {n}")
        try:
            row = row.split("\t")
            src_line, tgt_line = row[0].strip(), row[1].strip()
        except:
            breakpoint()
        # reach doc end
        if (src_line == "") and (tgt_line == ""):
            doc_id += 1
            continue
        # skip unaligned sentences
        if (src_line == "") or (tgt_line == ""):
            continue
        # dump sentences
        args.src_dump.write(src_line + "\n")
        args.tgt_dump.write(tgt_line + "\n")
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
    args.src_dump.close()
    args.tgt_dump.close()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        type=FileType("r"),
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
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100000,
        help="log every N lines",
    )
    args = parser.parse_args()
    main(args)
