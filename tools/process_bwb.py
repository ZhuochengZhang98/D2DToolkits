import os
import logging
import numpy as np
from argparse import ArgumentParser, Namespace


logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def process_train(train_path: str, dump_path: str, spliter: str, interval: int):
    # prepare files
    src = open(os.path.join(train_path, "train.zh"), "r")
    tgt = open(os.path.join(train_path, "train.en"), "r")
    src_dump = open(os.path.join(dump_path, "train.zh"), "w")
    tgt_dump = open(os.path.join(dump_path, "train.en"), "w")
    doc_ids = []

    # steam process
    for doc_id, (src_line, tgt_line) in enumerate(zip(src, tgt)):
        if doc_id % interval == 0:
            logger.info(f"Processing lines: {doc_id}")
        src_sents = src_line.strip().split(spliter)
        tgt_sents = tgt_line.strip().split(spliter)
        assert len(src_sents) == len(tgt_sents)
        for n, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
            # skip empty line
            if src_sent.strip() == "":
                continue
            if tgt_sent.strip() == "":
                continue
            src_dump.write(f"{src_sent.strip()}\n")
            tgt_dump.write(f"{tgt_sent.strip()}\n")
            doc_ids.append(doc_id)
    src_dump.close()
    tgt_dump.close()

    # post process
    docid_file = np.memmap(
        os.path.join(dump_path, "train.docid"),
        dtype=int,
        mode="w+",
        shape=(len(doc_ids),),
    )
    docid_file[:] = np.array(doc_ids)[:]
    logger.info(f"Processed sentences: {len(doc_ids)}")
    logger.info(f"Processed documents: {doc_id}")
    return


def process_test(test_path: str, dump_path: str, subset: str):
    # prepare files
    books = os.listdir(test_path)
    src_dump = open(os.path.join(dump_path, f"{subset}.zh"), "w")
    tgt_dump = open(os.path.join(dump_path, f"{subset}.en"), "w")
    doc_ids = []
    doc_id = 0
    for book in books:
        names = os.listdir(os.path.join(test_path, book))
        name_ids = set([i.split(".")[0] for i in names])
        for name_id in name_ids:
            src = os.path.join(test_path, book, f"{name_id}.chs_re.txt")
            tgt = os.path.join(test_path, book, f"{name_id}.ref_re.txt")
            if not (os.path.exists(src) and os.path.exists(tgt)):
                continue
            src = open(src, "r").readlines()
            tgt = open(tgt, "r").readlines()
            assert len(src) == len(tgt)
            for src_line, tgt_line in zip(src, tgt):
                src_dump.write(src_line.strip() + "\n")
                tgt_dump.write(tgt_line.strip() + "\n")
                doc_ids.append(doc_id)
            doc_id += 1
    src_dump.close()
    tgt_dump.close()

    # post process
    docid_file = np.memmap(
        os.path.join(dump_path, f"{subset}.docid"),
        dtype=int,
        mode="w+",
        shape=(len(doc_ids),),
    )
    docid_file[:] = np.array(doc_ids)[:]
    logger.info(f"Processed {subset} sentences: {len(doc_ids)}")
    logger.info(f"Processed {subset} documents: {doc_id}")
    return


def main(args):
    process_test(args.test_path, args.dump, "test")
    process_test(args.valid_path, args.dump, "valid")
    process_train(args.train_path, args.dump, args.spliter, 100000)
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="Used for BWB dataset")
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="path to raw source file.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        required=True,
        help="path to raw target file.",
    )
    parser.add_argument(
        "--valid-path",
        type=str,
        required=True,
        help="path to raw target file.",
    )
    parser.add_argument(
        "--dump",
        type=str,
        required=True,
        help="path to dump source sentences.",
    )
    parser.add_argument(
        "--spliter",
        type=str,
        default="<sep>",
        help="spliter to split the source document",
    )
    args = parser.parse_args()
    main(args)
