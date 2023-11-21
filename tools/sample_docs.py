from argparse import ArgumentParser, Namespace, FileType

import numpy as np


def main(args: Namespace):
    # sample docids
    docid = np.memmap(args.docid, dtype=int, mode="r")
    docid_u = np.arange(0, docid.max() + 1)
    choosed = set(np.random.choice(docid_u, args.sample_num, replace=False))

    # dump new data
    new_docids = []
    last_docid = docid[0]
    for src_line, tgt_line, docid in zip(args.src, args.tgt, docid):
        if docid in choosed:
            if len(new_docids) == 0:
                new_docids.append(0)
                last_docid = docid
            elif docid == last_docid:
                new_docids.append(new_docids[-1])
            else:
                new_docids.append(new_docids[-1] + 1)
                last_docid = docid
            args.src_dump.write(src_line)
            args.tgt_dump.write(tgt_line)
    args.src_dump.close()
    args.tgt_dump.close()

    # dump new docids
    new_docids = np.array(new_docids)
    docid_dump = np.memmap(
        args.docid_dump,
        dtype=int,
        mode="w+",
        shape=(len(new_docids),),
    )
    docid_dump[:] = new_docids[:]
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src",
        required=True,
        type=FileType("r"),
        help="path to source file",
    )
    parser.add_argument(
        "--tgt",
        required=True,
        type=FileType("r"),
        help="path to target file",
    )
    parser.add_argument(
        "--docid",
        required=True,
        type=str,
        help="path to docid file",
    )
    parser.add_argument(
        "--sample-num",
        type=int,
        required=True,
        help="number of sampled docs",
    )
    parser.add_argument(
        "--src-dump",
        required=True,
        type=FileType("w"),
        help="path to source file",
    )
    parser.add_argument(
        "--tgt-dump",
        required=True,
        type=FileType("w"),
        help="path to target file",
    )
    parser.add_argument(
        "--docid-dump",
        required=True,
        type=str,
        help="path to docid file",
    )
    args = parser.parse_args()
    main(args)
