import logging
from argparse import ArgumentParser, Namespace, FileType

import numpy as np

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def filter(args, adq_score, cs_lang_score, en_lang_score):
    if args.adq_score is not None:
        if adq_score < args.adq_score:
            return True
    if args.lang_score is not None:
        if cs_lang_score < args.lang_score:
            return True
        if en_lang_score < args.lang_score:
            return True
    return False


def main(args: Namespace):
    # steam process
    doc_ids = []
    docid = 0
    for n, line in enumerate(args.data):
        if n % 100000 == 0:
            logger.info(f"Processing lines: {n}")
        if line.strip() == "":
            docid += 1
            continue
        line_info = line.strip().split("\t")
        sent_id, adq_score, cs_lang_score, en_lang_score, src, tgt = line_info
        if filter(args, adq_score, cs_lang_score, en_lang_score):
            continue
        args.src_dump.write(src.strip() + "\n")
        args.tgt_dump.write(tgt.strip() + "\n")
        doc_ids.append(docid)
    args.src_dump.close()
    args.tgt_dump.close()

    # post process
    docid_file = np.memmap(
        args.docid_dump,
        dtype=int,
        mode="w+",
        shape=(len(doc_ids),),
    )
    docid_file[:] = np.array(doc_ids)[:]
    logger.info(f"Processed lines: {n}")
    logger.info(f"Processed documents: {doc_ids[-1]}")
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
        "--lang-score",
        default=None,
        type=float,
        help="minimum language score",
    )
    parser.add_argument(
        "--adq-score",
        default=None,
        type=float,
        help="minimum cross entropy score",
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
