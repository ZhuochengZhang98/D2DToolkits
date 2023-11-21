import os
import re
import json
import logging
from argparse import ArgumentParser, FileType, Namespace
from functools import partial
from typing import List, Tuple

import numpy as np
from sacrebleu import BLEU, CHRF
from sacrebleu.significance import PairedTest
from sacrebleu.utils import print_results_table


logger = logging.getLogger(__name__)


def preprocess_seg(
    segs: List[str],
    docids: List[int],
    spliter: str,
    combine: bool = True,
    old: bool = False,
) -> Tuple[List[str], List[str]]:
    sents = []
    for seg in segs:
        if old:
            sents.extend(seg.strip().split(spliter))
        else:
            seg = re.sub(f"{spliter}$", "", seg.strip())  # remove last spliter
            sents.extend(seg.split(spliter))
    sents = [i.strip() for i in sents]

    assert len(sents) == len(
        docids
    ), f"Sentence number mismatch: {len(sents)} vs {len(docids)}"

    docs = {i: [] for i in set(docids)}
    for sent, docid in zip(sents, docids):
        docs[docid].append(sent)
    if combine:
        docs = [" ".join(i) for i in docs.values()]
    else:
        return [i for i in docs.values()]
    return docs, sents


def evaluate_comet(ref_sents, hyp_sents, src_sents):
    from comet import download_model, load_from_checkpoint

    data = [
        {"src": i, "mt": j, "ref": k}
        for i, j, k in zip(src_sents, hyp_sents, ref_sents)
    ]
    if args.comet_model is None:
        model_path = download_model("Unbabel/wmt22-comet-da")
    else:
        model_path = args.comet_model
    model = load_from_checkpoint(model_path)
    return model.predict(data, batch_size=32, gpus=1)


def main(args: Namespace):
    # for compatibility
    process_seg = partial(preprocess_seg, old=args.old, spliter=args.spliter)

    # prepare refs & hyps
    docids = np.memmap(args.docid, dtype=int, mode="r")
    refs_raw = [i.readlines() for i in args.ref]
    hyps_raw = [i.readlines() for i in args.sys]
    refs = [process_seg(i, docids) for i in refs_raw]
    hyps = [process_seg(i, docids) for i in hyps_raw]
    names = [i.name for i in args.sys]
    refs_doc = [i[0] for i in refs]
    refs_sent = [i[1] for i in refs]
    hyps_doc = [i[0] for i in hyps]
    hyps_sent = [i[1] for i in hyps]

    if args.src is not None:
        src_raw = args.src.readlines()
        _, src_sents = process_seg(src_raw, docids)

    if args.comet:
        assert len(hyps_sent) == 1
        assert len(refs_sent) == 1
        scores = evaluate_comet(refs_sent[0], hyps_sent[0], src_sents)
        for n, score in enumerate(scores["scores"]):
            print(f"sentence\t{n}\t{score}")
        print(f"system\t{scores['system_score']}")
        return

    # prepare metrics
    lc = args.case_insensitive
    bleu_doc = BLEU(trg_lang=args.lang, references=refs_doc, lowercase=lc)
    bleu_sent = BLEU(trg_lang=args.lang, references=refs_sent, lowercase=lc)
    chrf_sent = CHRF(lowercase=lc, references=refs_sent)
    chrf_doc = CHRF(lowercase=lc, references=refs_doc)

    # calculate blonde score for zhen
    if args.lang == "en":
        from blonde import BLONDE

        refs = [process_seg(i, docids, combine=False) for i in refs_raw]
        hyps = [process_seg(i, docids, combine=False) for i in hyps_raw]
        blonde = BLONDE(references=refs, lowercase=lc)
    else:
        blonde = None
        hyps = [None for i in hyps_doc]

    # perform significant test
    if args.significance:
        assert len(args.sys) > 1, "systems not enough"
        ps = PairedTest(
            list(zip(names, hyps_sent)), {"bleu_doc": bleu_sent}, None, "bs", 1000
        )
        sigs, scores = ps()
        print_results_table(scores, sigs, args)
    else:
        # sig_doc = bleu_doc.get_signature()
        # sig_sent = bleu_sent.get_signature()
        # sig_chrf = chrf.get_signature()
        scores = {}
        for name, hyp_sent, hyp_doc, hyp in zip(names, hyps_sent, hyps_doc, hyps):
            scores[name] = {
                "s-BLEU": str(bleu_sent.corpus_score(hyp_sent, None)),
                "d-BLEU": str(bleu_doc.corpus_score(hyp_doc, None)),
                "s-ChrF": str(chrf_sent.corpus_score(hyp_sent, None)),
                "d-ChrF": str(chrf_doc.corpus_score(hyp_doc, None)),
            }
            if blonde is not None:
                bl_score = blonde.corpus_score(hyp)
                scores[name]["BLONDE"] = f"BLONDE = {bl_score.score:.4f}"
        print(name)
        print(json.dumps(scores[name], indent=4))
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ref",
        "-r",
        nargs="+",
        type=FileType("r"),
        required=True,
        help="Path(s) to reference",
    )
    parser.add_argument(
        "--sys",
        "-s",
        nargs="+",
        type=FileType("r"),
        required=True,
        help="Path(s) to system hypothesis",
    )
    parser.add_argument(
        "--src",
        type=FileType("r"),
        default=None,
        help="Path(s) to source",
    )
    parser.add_argument(
        "--comet",
        default=False,
        action="store_true",
        help="Whether to use comet score(will early exit)",
    )
    parser.add_argument(
        "--comet-model",
        default=None,
        type=str,
        help="Path to comet model",
    )
    parser.add_argument(
        "--docid",
        "-d",
        type=str,
        required=True,
        help="Path to reference docid file",
    )
    parser.add_argument(
        "--spliter",
        type=str,
        required=True,
        help="The spliter",
    )
    parser.add_argument(
        "--significance",
        "-sig",
        action="store_true",
        default=False,
        help="Calculate significance via paired bootstrap resampling. "
        "The first system is seen as baseline system",
    )
    parser.add_argument(
        "--case-insensitive",
        "-c",
        action="store_true",
        default=False,
        help="apply case insensitive BLEU",
    )
    parser.add_argument(
        "--lang",
        "-l",
        required=True,
        type=str,
        help="Target language",
    )
    parser.add_argument(
        "--old",
        action="store_true",
        default=False,
        help="Use old fashion",
    )
    args = parser.parse_args()
    args.format = "json"
    if args.comet:
        assert args.src is not None, "--src is required for comet score"
    main(args)
