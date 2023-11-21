import json
import os
import logging
import random as rd
from argparse import ArgumentParser, FileType, Namespace
from typing import List
from multiprocessing import Pool
from functools import partial

import numpy as np
import sacremoses as sm
import sentencepiece as spm
from subword_nmt.apply_bpe import BPE


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_context(
    src_doc: List[str],
    tgt_doc: List[str],
    src_lengths: List[int],
    tgt_lengths: List[int],
    line_id: int,
    segment_length: int = None,
    context_num: int = None,
    sample_length: bool = False,
):
    end_id = max(0, line_id - context_num) if context_num is not None else 0
    i = line_id - 1
    for i in range(line_id - 1, end_id, -1):
        # sample the segment length
        if sample_length and (segment_length is not None):
            max_len = rd.randint(1, segment_length)
        else:
            max_len = segment_length

        # collect the context
        if max_len:
            src_len = sum(src_lengths[i : line_id + 1]) + (line_id - i)
            if src_len >= max_len:
                break
            tgt_len = sum(tgt_lengths[i : line_id + 1]) + (line_id - i)
            if tgt_len >= max_len:
                break
    # at least one sentence is included
    if sample_length and (segment_length is not None):
        i = min(i, line_id - 2)
    src_contexts = src_doc[i + 1 : line_id]
    tgt_contexts = tgt_doc[i + 1 : line_id]
    return src_contexts, tgt_contexts


def process_line(
    line: str,
    bpe: BPE = None,
    tok: sm.MosesTokenizer = None,
    sp: spm.SentencePieceProcessor = None,
):
    line = line.strip()
    if tok is not None:
        line = tok.tokenize(line, return_str=True)
    if bpe is not None:
        line = bpe.process_line(line)
    if sp is not None:
        line = " ".join(sp.encode_as_pieces(line))
    return line


def process_lines(
    lines: List[str],
    bpe: BPE = None,
    tok: sm.MosesTokenizer = None,
    sp: spm.SentencePieceProcessor = None,
):
    return [process_line(line, bpe, tok, sp) for line in lines]


def get_lens(lines: List[str]):
    return [len(line.split()) for line in lines]


def main(args: Namespace):
    # prepare processor
    if args.bpe_path is not None:
        bpe = BPE(open(args.bpe_path, "r"))
        tok = sm.MosesTokenizer(lang=args.tgt_lang)
        sp = None
    else:
        bpe = None
        tok = None
        sp = spm.SentencePieceProcessor(model_file=args.spm_path)

    # prepare function
    tok_line = partial(process_line, bpe=bpe, tok=tok, sp=sp)
    tok_lines = partial(process_lines, bpe=bpe, tok=tok, sp=sp)

    # prepare data
    logger.info("Preparing data...")
    contra_data = json.load(args.json)
    documents = {}
    lengths = {}
    paths = os.listdir(args.data_path)
    pool = Pool(args.num_workers)
    for n, year in enumerate(paths):
        logger.info(f"Finished {n}\t Total {len(paths)}\tProcessing {year}...")
        file_names = os.listdir(os.path.join(args.data_path, year))
        file_paths = [os.path.join(args.data_path, year, i) for i in file_names]
        file_lines = [open(i, "r").readlines() for i in file_paths]
        docs = pool.map(tok_lines, file_lines)
        lens = pool.map(get_lens, docs)
        documents.update(
            {f"{file_names[i]}": docs[i] for i in range(len(file_names))}
        )
        lengths.update(
            {f"{file_names[i]}": lens[i] for i in range(len(file_names))}
        )
    pool.close()

    # extract data
    logger.info("Extracting data...")
    src_output = open(os.path.join(args.output, f"contrapro.{args.src_lang}"), "w")
    tgt_output = open(os.path.join(args.output, f"contrapro.{args.tgt_lang}"), "w")
    docids = []
    docid = 0
    n_lines = 0
    for n, data in enumerate(contra_data):
        if args.sample_num is not None:
            if n >= args.sample_num:
                break
        if n % 100 == 0:
            logger.info(f"Finished {n}\t Total {len(contra_data)}")
        doc_name = data["document id"].split(".")[0]
        line_id = data["segment id"] - 1
        src_line = documents[f"{doc_name}.{args.src_lang}"][line_id]
        tgt_lines = [documents[f"{doc_name}.{args.tgt_lang}"][line_id]]
        if not args.no_error_line:
            error_lines = [i["contrastive"] for i in data["errors"]]
            for line in error_lines:
                line = tok_line(line)
                tgt_lines.append(line)
        src_contexts, tgt_contexts = get_context(
            documents[f"{doc_name}.{args.src_lang}"],
            documents[f"{doc_name}.{args.tgt_lang}"],
            lengths[f"{doc_name}.{args.src_lang}"],
            lengths[f"{doc_name}.{args.tgt_lang}"],
            line_id,
            args.segment_length,
            args.context_num,
            args.sample_length,
        )
        # write to file
        for tgt_line in tgt_lines:
            docids.extend([docid] * (1 + len(tgt_contexts)))
            docid += 1
            for src_ctx, tgt_ctx in zip(src_contexts, tgt_contexts):
                src_output.write(src_ctx.strip() + "\n")
                tgt_output.write(tgt_ctx.strip() + "\n")
                n_lines += 1
            src_output.write(src_line.strip() + "\n")
            tgt_output.write(tgt_line.strip() + "\n")
            n_lines += 1
    assert len(docids) == n_lines

    # write docid
    docids = np.array(docids, dtype=int)
    docid_file = np.memmap(
        os.path.join(args.output, f"contrapro.docid"),
        dtype=int,
        mode="w+",
        shape=(len(docids),),
    )
    docid_file[:] = docids[:]
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--json",
        type=FileType("r"),
        required=True,
        help="json file to load",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="path to tokenized opensubtitle documents",
    )
    parser.add_argument(
        "--bpe-path",
        default=None,
        type=str,
        help="path to bpe model",
    )
    parser.add_argument(
        "--spm-path",
        default=None,
        type=str,
        help="path to spm model",
    )
    parser.add_argument(
        "--src-lang",
        default="en",
        type=str,
        help="source language",
    )
    parser.add_argument(
        "--tgt-lang",
        default="de",
        type=str,
        help="target language",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=None,
        help="segment length",
    )
    parser.add_argument(
        "--sample-length",
        default=False,
        action="store_true",
        help="sample segment length",
    )
    parser.add_argument(
        "--sample-num",
        default=None,
        type=int,
        help="sample number",
    )
    parser.add_argument(
        "--no-error-line",
        default=False,
        action="store_true",
        help="do not include error line",
    )
    parser.add_argument(
        "--context-num",
        type=int,
        default=None,
        help="number of context sentences",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="number of workers",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to output file",
    )
    args = parser.parse_args()
    if (args.bpe_path is None) and (args.spm_path is None):
        raise ValueError("Either bpe_path or spm_path must be provided.")
    main(args)
