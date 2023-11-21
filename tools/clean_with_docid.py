# This script is used to clean the data with docid file at document-level.
import logging
import math
import sys
from abc import ABC
from argparse import ArgumentParser, FileType, Namespace
from io import TextIOWrapper
from tempfile import TemporaryFile
from typing import DefaultDict, List, Tuple, Union

import gcld3
import numpy as np


formater = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formater)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class DocReader:
    def __init__(
        self,
        src_file: Union[str, TextIOWrapper],
        tgt_file: Union[str, TextIOWrapper],
        docid_file: str,
    ):
        if isinstance(src_file, str):
            src_file = open(src_file, "r")
        else:
            assert isinstance(src_file, TextIOWrapper)
        if isinstance(tgt_file, str):
            tgt_file = open(tgt_file, "r")
        else:
            assert isinstance(tgt_file, TextIOWrapper)
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.docid_file = np.memmap(docid_file, int, mode="r")
        return

    def __iter__(self) -> Tuple[List[str], List[str]]:
        last_docid = None
        src_doc = []
        tgt_doc = []
        for src_sent, tgt_sent, docid in zip(
            self.src_file,
            self.tgt_file,
            self.docid_file,
        ):
            if last_docid is None:
                last_docid = docid
                continue
            if last_docid != docid:
                yield src_doc, tgt_doc
                src_doc = []
                tgt_doc = []
                last_docid = docid
            src_doc.append(src_sent)
            tgt_doc.append(tgt_sent)

    def __len__(self):
        return len(set(self.docid_file))


class DocDumper:
    def __init__(self, f: TextIOWrapper):
        self.__f__ = f
        return

    def dump(self, doc: List[str]):
        for line in doc:
            self.__f__.write(line.rstrip() + "\n")
        return


class Cleaner(ABC):
    name = "Cleaner"

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        return

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        return parser

    def __call__(self, src_doc: List[str], tgt_doc: List[str]) -> bool:
        raise NotImplementedError


class Register(dict):
    def __init__(self):
        super().__init__()
        self.__main_names__ = []
        self.__short_name__ = []

    def __call__(self, *short_names: str):
        def registe_processor(processor: Cleaner):
            main_name = str(processor).split(".")[-1][:-2]
            names = [main_name]
            names.extend(short_names)
            for name in names:
                assert self.get(name, None) is None, "Processor Name Conflict %s" % name
                self.__setitem__(name, processor)
            self.__main_names__.append(main_name)
            self.__short_name__.extend(short_names)
            processor.name = main_name
            return processor

        return registe_processor

    def __iter__(self):
        return self.__main_names__.__iter__()

    @property
    def names(self):
        return self.__short_name__


PROCESSORS = Register()


@PROCESSORS("length")
class LengthCleaner(Cleaner):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.max_len = getattr(args, "max_length", math.inf)
        self.min_len = getattr(args, "min_length", 1)
        ratio = getattr(args, "length_ratio", math.inf)
        self.max_ratio = ratio if ratio > 1 else 1 / ratio
        self.min_ratio = ratio if ratio < 1 else 1 / ratio
        return

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument(
            "--length-ratio",
            type=float,
            default=math.inf,
            help="max length ratio",
        )
        parser.add_argument(
            "--min-length",
            default=1,
            type=int,
            help="min length",
        )
        parser.add_argument(
            "--max-length",
            default=math.inf,
            type=int,
            help="max length",
        )
        return super().add_args(parser)

    def __call__(self, src_doc: List[str], tgt_doc: List[str]) -> bool:
        src_len = sum([len(line.split()) for line in src_doc])
        tgt_len = sum([len(line.split()) for line in tgt_doc])
        if src_len > self.max_len:
            return False
        if tgt_len > self.max_len:
            return False
        if src_len < self.min_len:
            return False
        if tgt_len < self.min_len:
            return False
        if (src_len / tgt_len) < self.min_ratio:
            return False
        if (src_len / tgt_len) > self.max_ratio:
            return False
        return True


@PROCESSORS("sent_length")
class SentenceLengthCleaner(Cleaner):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.max_len = getattr(args, "sent_max_length", math.inf)
        self.min_len = getattr(args, "sent_min_length", 1)
        ratio = getattr(args, "sent_length_ratio", math.inf)
        self.max_ratio = ratio if ratio > 1 else 1 / ratio
        self.min_ratio = ratio if ratio < 1 else 1 / ratio
        return

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument(
            "--sent-length-ratio",
            type=float,
            default=math.inf,
            help="max length ratio",
        )
        parser.add_argument(
            "--sent-min-length",
            default=1,
            type=int,
            help="min length",
        )
        parser.add_argument(
            "--sent-max-length",
            default=math.inf,
            type=int,
            help="max length",
        )
        return super().add_args(parser)

    def __call__(self, src_doc: List[str], tgt_doc: List[str]) -> bool:
        for src_line, tgt_line in src_doc, tgt_doc:
            src_len = len(src_line.split())
            tgt_len = len(tgt_line.split())
            if src_len > self.max_len:
                return False
            if tgt_len > self.max_len:
                return False
            if src_len < self.min_len:
                return False
            if tgt_len < self.min_len:
                return False
            if (src_len / tgt_len) < self.min_ratio:
                return False
            if (src_len / tgt_len) > self.max_ratio:
                return False
        return True


@PROCESSORS("lang")
class LangCleaner(Cleaner):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang
        self.detector = gcld3.NNetLanguageIdentifier(
            min_num_bytes=0, max_num_bytes=1000
        )
        if args.detokenizer == "bpe":
            self.detok = lambda x: x.replace("@@ ", "")
        elif args.detokenizer == "spm":
            self.detok = lambda x: x.replace(" ", "").replace("▁", " ").strip()
        else:
            self.detok = lambda x: x
        return

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument(
            "--src-lang",
            type=str,
            default=None,
            help="source language",
        )
        parser.add_argument(
            "--tgt-lang",
            type=str,
            default=None,
            help="target language",
        )
        parser.add_argument(
            "--detokenizer",
            type=str,
            default=None,
            choices=["bpe", "spm"],
            help="tokenizer used to detokenize the text",
        )
        return super().add_args(parser)

    def __call__(self, src_doc: str, tgt_doc: str) -> bool:
        src_langs = DefaultDict(list)
        tgt_langs = DefaultDict(list)
        for src_line, tgt_line in zip(src_doc, tgt_doc):
            src_line = self.detok(src_line)
            tgt_line = self.detok(tgt_line)
            src_lang = self.detector.FindLanguage(text=src_line)
            tgt_lang = self.detector.FindLanguage(text=tgt_line)
            src_langs[src_lang.language].append(src_lang.probability)
            tgt_langs[tgt_lang.language].append(tgt_lang.probability)
        src_lang = max(src_langs, key=lambda x: sum(src_langs[x]))
        tgt_lang = max(tgt_langs, key=lambda x: sum(tgt_langs[x]))
        if src_lang != self.src_lang:
            return False
        if tgt_lang != self.tgt_lang:
            return False
        return True


@PROCESSORS("dedup")
class DeDuplicationCleaner(Cleaner):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.src_set = set()
        self.tgt_set = set()
        return

    def __call__(self, src_doc: str, tgt_doc: str) -> bool:
        src_str = "".join(src_doc)
        src_str = src_str.replace(" ", "")
        tgt_str = "".join(tgt_doc)
        tgt_str = tgt_str.replace(" ", "")
        if src_str in self.src_set:
            return False
        if tgt_str in self.tgt_set:
            return False
        self.src_set.add(src_str)
        self.tgt_set.add(tgt_str)
        return True


class CleanPipeline:
    def __init__(self, args: Namespace) -> None:
        self.cleaners = []
        for cleaner in args.processors:
            self.cleaners.append(PROCESSORS[cleaner](args))
        self.counter = DefaultDict(int)
        return

    def __call__(self, src_doc: str, tgt_doc: str) -> bool:
        self.counter["total"] += 1
        accept = True
        for cleaner in self.cleaners:
            if not cleaner(src_doc, tgt_doc):
                self.counter["discarded"] += 1
                self.counter[cleaner.name] += 1
                accept = False
                break
        if self.counter["total"] % 1000 == 0:
            log_info = ", ".join(
                [
                    f"{cleaner.name}: {self.counter[cleaner.name]}"
                    for cleaner in self.cleaners
                ]
            )
            logger.info(
                f"Processed {self.counter['total']} documents, "
                f"discarded {self.counter['discarded']} documents, "
                f"{log_info}"
            )
        return accept

    def info(self):
        log_info = ", ".join(
            [
                f"{cleaner.name}: {self.counter[cleaner.name]}"
                for cleaner in self.cleaners
            ]
        )
        logger.info(
            f"Processed {self.counter['total']} documents, "
            f"discarded {self.counter['discarded']} documents, "
            f"{log_info}"
        )
        return
    
    def clean_counter(self):
        self.counter = DefaultDict(int)
        return


def main(args: Namespace):
    # prepare cleaner
    logger.info(f"Clean Processors: {args.processors}")
    doc_reader = DocReader(args.src, args.tgt, args.docid)
    src_dumper = DocDumper(args.src_dump)
    tgt_dumper = DocDumper(args.tgt_dump)
    cleaner = CleanPipeline(args)

    # clean
    logger.info(f"Total documents: {len(doc_reader)}")
    discarded = 0
    new_docids = [-1]
    for src_doc, tgt_doc in doc_reader:
        if cleaner(src_doc, tgt_doc):
            new_docids.extend([new_docids[-1] + 1] * len(src_doc))
            src_dumper.dump(src_doc)
            tgt_dumper.dump(tgt_doc)
        else:
            discarded += 1
    new_docids.pop(0)
    new_docids = np.array(new_docids)

    # dump docids
    cleaner.info()
    if len(new_docids) <= 0:
        logger.warning("No documents left after cleaning, skip dumping docids")
        return
    docids_f = np.memmap(
        filename=args.docid_dump,
        dtype=int,
        mode="w+",
        shape=len(new_docids),
    )
    docids_f[:] = np.array(new_docids)[:]
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    # basic IO arguments
    parser.add_argument(
        "--src",
        type=FileType("r"),
        required=True,
        help="path to source file",
    )
    parser.add_argument(
        "--tgt",
        type=FileType("r"),
        required=True,
        help="path to target file",
    )
    parser.add_argument(
        "--docid",
        type=str,
        required=True,
        help="path to docid file",
    )
    parser.add_argument(
        "--src-dump",
        type=FileType("w"),
        default=TemporaryFile("w"),
        help="path to dump source file. Default: TemporaryFile",
    )
    parser.add_argument(
        "--tgt-dump",
        type=FileType("w"),
        default=TemporaryFile("w"),
        help="path to dump target file. Default: TemporaryFile",
    )
    parser.add_argument(
        "--docid-dump",
        type=str,
        default="/dev/zero",
        help="path to dump docid file. Default: /dev/zero",
    )
    # cleaning arguments
    parser.add_argument(
        "--processors",
        type=str,
        nargs="+",
        choices=PROCESSORS.names,
        default=[],
    )
    # logging arguments
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="log interval",
    )
    # add arguments for each processor
    for cleaner in PROCESSORS:
        parser = PROCESSORS[cleaner].add_args(parser)
    args = parser.parse_args()
    main(args)
