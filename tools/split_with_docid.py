import logging
import os
from argparse import ArgumentParser, Namespace

import numpy as np

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def sample_indices(max_size: int, valid_size: int, test_size: int = None):
    valid_doc_indices = np.random.choice(max_size, size=valid_size)
    test_doc_indices = []
    repeat = test_size is not None
    # re-sample till no overlap
    while repeat:
        repeat = False
        test_doc_indices = np.random.choice(max_size, size=test_size)
        for i in test_doc_indices:
            if i in valid_doc_indices:
                repeat = True
                break
    return valid_doc_indices, test_doc_indices


def main(args: Namespace):
    src_lang = args.src.split(".")[-1]
    tgt_lang = args.tgt.split(".")[-1]
    docids = np.memmap(args.docid, int, "r")
    src = open(args.src, "r")
    tgt = open(args.tgt, "r")

    # sample doc indices
    valid_doc_indices, test_doc_indices = sample_indices(
        docids.max(), args.valid_size, args.test_size
    )

    # prepare dump
    train_src = open(os.path.join(args.dump, f"train.{src_lang}"), "w")
    train_tgt = open(os.path.join(args.dump, f"train.{tgt_lang}"), "w")
    valid_src = open(os.path.join(args.dump, f"valid.{src_lang}"), "w")
    valid_tgt = open(os.path.join(args.dump, f"valid.{tgt_lang}"), "w")
    if len(test_doc_indices) > 0:
        test_src = open(os.path.join(args.dump, f"test.{src_lang}"), "w")
        test_tgt = open(os.path.join(args.dump, f"test.{tgt_lang}"), "w")
    train_docids = []
    valid_docids = []
    test_docids = []

    # dump text
    last_docid = None
    for n, (docid, src_line, tgt_line) in enumerate(zip(docids, src, tgt)):
        if n % 100000 == 0:
            logger.info(f"Processing lines: {n}")

        if docid in valid_doc_indices:
            if len(valid_docids) == 0:
                valid_docids.append(0)
            elif last_docid != docid:
                valid_docids.append(valid_docids[-1] + 1)
            else:
                valid_docids.append(valid_docids[-1])
            valid_src.write(src_line)
            valid_tgt.write(tgt_line)
        elif docid in test_doc_indices:
            if len(test_docids) == 0:
                test_docids.append(0)
            elif last_docid != docid:
                test_docids.append(test_docids[-1] + 1)
            else:
                test_docids.append(test_docids[-1])
            test_src.write(src_line)
            test_tgt.write(tgt_line)
        else:
            if len(train_docids) == 0:
                train_docids.append(0)
            elif last_docid != docid:
                train_docids.append(train_docids[-1] + 1)
            else:
                train_docids.append(train_docids[-1])
            train_src.write(src_line)
            train_tgt.write(tgt_line)
        last_docid = docid
    
    # test dump
    if test_doc_indices is not None:
        test_docid_file = np.memmap(
            os.path.join(args.dump, "test.docid"),
            dtype=int,
            mode="w+",
            shape=(len(test_docids),),
        )
        test_docid_file[:] = np.array(test_docids)[:]
        logger.info(f"Test lines: {len(test_docids)}")
        logger.info(f"Test documents: {test_docids[-1]}")
        test_src.close()
        test_tgt.close()

    # train/valid set dump
    train_docid_file = np.memmap(
        os.path.join(args.dump, "train.docid"),
        dtype=int,
        mode="w+",
        shape=(len(train_docids),),
    )
    valid_docid_file = np.memmap(
        os.path.join(args.dump, "valid.docid"),
        dtype=int,
        mode="w+",
        shape=(len(valid_docids),),
    )

    train_docid_file[:] = np.array(train_docids)[:]
    valid_docid_file[:] = np.array(valid_docids)[:]
    logger.info(f"Train lines: {len(train_docids)}")
    logger.info(f"Train documents: {train_docids[-1]}")
    logger.info(f"Valid lines: {len(valid_docids)}")
    logger.info(f"Valid documents: {valid_docids[-1]}")
    train_src.close()
    train_tgt.close()
    valid_src.close()
    valid_tgt.close()
    
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
        "--docid",
        type=str,
        required=True,
        help="path to raw source file.",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=10,
        help="valid set docs",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="test set docs",
    )
    parser.add_argument(
        "--dump",
        type=str,
        required=True,
        help="path to dump source sentences.",
    )
    args = parser.parse_args()
    main(args)
