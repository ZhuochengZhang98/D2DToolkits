from argparse import ArgumentParser, Namespace

import numpy as np


def make_doc(src_lines, tgt_line, doc_ids):
    docs = {i:[] for i in doc_ids}
    for src_line, tgt_line, doc_id in zip(src_lines, tgt_line, doc_ids):
        docs[doc_id].append((src_line, tgt_line))
    return docs


def check_doc(docs):
    for docid in docs:
        actual_id = eval(docs[docid][0][0].split()[-1])
        for src_line, tgt_line in docs[docid]:
            src_id = eval(src_line.split()[-1])
            tgt_id = eval(tgt_line.split()[-1])
            if (src_id != actual_id) or (tgt_id != actual_id):
                raise ValueError(f"src_id: {src_id}, tgt_id: {tgt_id}")
    return


def main(args: Namespace):
    src_lines = open(args.dummy_src, "r").readlines()
    tgt_lines = open(args.dummy_tgt, "r").readlines()
    doc_ids = np.memmap(args.dummy_docid, dtype=int, mode="r")

    docs = make_doc(src_lines, tgt_lines, doc_ids)
    check_doc(docs)
    print("DUMMY CHECK DONE")
    print("NO ERROR FOUND")
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dummy-src",
        type=str,
        required=True,
        help="path to dump source",
    )
    parser.add_argument(
        "--dummy-tgt",
        type=str,
        required=True,
        help="path to dump target",
    )
    parser.add_argument(
        "--dummy-docid",
        type=str,
        required=True,
        help="path to dump docid file",
    )
    args = parser.parse_args()
    main(args)
