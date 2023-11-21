import os
import random
from argparse import ArgumentParser, Namespace


def main(args: Namespace):
    doc_id_file = open(os.path.join(args.dump, "dummy.docid"), "w")
    src_file = open(os.path.join(args.dump, "dummy.src"), "w")
    tgt_file = open(os.path.join(args.dump, "dummy.tgt"), "w")
    docid = 0
    for i in range(args.length):
        # 1% chance of new doc
        if random.random() < 0.01:
            docid += 1
        doc_id_file.write(f"{docid}\t{docid}\n")
        # 0.5% chance of too short line
        if random.random() > 0.995:
            src_file.write(f"{docid}\n")
            tgt_file.write(f"{docid}\n")
        else:
            src_file.write(f"src line {i} {docid}\n")
            tgt_file.write(f"tgt line {i} {docid}\n")
    doc_id_file.close()
    src_file.close()
    tgt_file.close()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--length",
        type=int,
        default=10000,
        help="length of dummy file",
    )
    parser.add_argument(
        "--dump",
        type=str,
        required=True,
        help="path to dump files",
    )
    args = parser.parse_args()
    main(args)
