import logging
import os

import torch
from fairseq import utils, search
from fairseq.data import Dictionary, data_utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.sequence_scorer import SequenceScorer
from fairseq.sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment

from .document_dataset import DocumentDataset
from .document_config import DocumentConfig
from .segment_generator import SegmentGenerator
from .utils import is_ampere, FinetunedDictionary


logger = logging.getLogger(__name__)


@register_task("document_translation", dataclass=DocumentConfig)
class DocumentTranslation(TranslationTask):
    cfg: DocumentConfig

    def __init__(self, cfg: DocumentConfig, src_dict: Dictionary, tgt_dict: Dictionary):
        super().__init__(cfg, src_dict, tgt_dict)
        self.finetune_mbart = cfg.finetune_mbart
        self.model = None
        return

    def load_dataset(
        self,
        split: str,
        epoch: int = 1,
        combine: bool = False,
        **kwargs,
    ):
        # prepare path
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]
        # load data
        self.datasets[split] = DocumentDataset(
            data_path=data_path,
            split=split,
            src=self.cfg.source_lang,
            tgt=self.cfg.target_lang,
            src_dict=self.src_dict,
            tgt_dict=self.tgt_dict,
            combine=combine,
            cfg=self.cfg,
            model_cfg=self.model_cfg,
        )
        return

    @classmethod
    def setup_task(cls, cfg: DocumentConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = FinetunedDictionary.load(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = FinetunedDictionary.load(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )

        # enable mBART
        if cfg.finetune_mbart:
            assert cfg.langs is not None
            langs = cfg.langs.split(",")
            assert not cfg.use_tags, "mBART does not support tags"
            assert not cfg.use_mask, "mBART does not use attention mask"
            for dic in [src_dict, tgt_dict]:
                for l in langs:
                    dic.add_symbol("[{}]".format(l))
                dic.add_symbol("<mask>")
        if cfg.word_drop > 0:
            src_dict.add_symbol(cfg.mask_token)
            tgt_dict.add_symbol(cfg.mask_token)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        # enable TF32
        if is_ampere():
            logger.info("Enable pytorch TF32")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return cls(cfg, src_dict, tgt_dict)

    def build_model(self, cfg: DocumentConfig, from_checkpoint=False):
        self.model_cfg = cfg
        self.model = super().build_model(cfg, from_checkpoint)
        return self.model

    @property
    def src_spliter(self) -> int:
        return self.src_dict.eos_index

    @property
    def tgt_spliter(self) -> int:
        return self.tgt_dict.eos_index

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        if getattr(args, "score_reference", False):
            if self.finetune_mbart:
                return SequenceScorer(
                    self.target_dictionary,
                    compute_alignment=getattr(args, "print_alignment", False),
                    eos=self.tgt_dict.index("[{}]".format(self.cfg.target_lang)),
                )
            else:
                return SequenceScorer(
                    self.target_dictionary,
                    compute_alignment=getattr(args, "print_alignment", False),
                )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if self.cfg.force_decode:
                seq_gen_cls = SegmentGenerator
                extra_gen_cls_kwargs.update(
                    {
                        "src_spliter": self.src_spliter,
                        "tgt_spliter": self.tgt_spliter,
                        "disable_incremental": self.cfg.disable_incremental,
                        "generate_with_golden": self.cfg.golden_context,
                        "slide_decode": self.cfg.slide_decode,
                        "context_window": self.cfg.context_window,
                    }
                )
            elif getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        if self.finetune_mbart:
            extra_gen_cls_kwargs["eos"] = self.tgt_dict.index(
                "[{}]".format(self.cfg.target_lang)
            )

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        if self.cfg.allow_longer:
            logger.info("Longger sentences are allowed")
            return indices
        return super().filter_indices_by_size(
            indices, dataset, max_positions, ignore_invalid_inputs
        )

    def begin_epoch(self, epoch, model):
        if (self.cfg.profile_model) and (epoch == 1):
            self.profile(model)
        if hasattr(model, "set_seg_length"):
            src_seg_length = self.dataset("train").src_sizes.mean()
            tgt_seg_length = self.dataset("train").tgt_sizes.mean()
            model.set_seg_length(src_seg_length, tgt_seg_length)
        return super().begin_epoch(epoch, model)

    @torch.no_grad()
    def profile(self, model):
        from deepspeed.profiling.flops_profiler import get_model_profile
        from deepspeed.accelerator import get_accelerator

        model_inputs = self.dataset("train").get_dummy_batch(self.cfg.segment_length)
        model_inputs = utils.move_to_cuda(model_inputs, "cuda:0")
        with get_accelerator().device(0):
            flops, macs, params = get_model_profile(
                model,
                kwargs=model_inputs,
                print_profile=True,
                detailed=True,
            )
        return
