import logging
import os
from collections import defaultdict
from typing import Tuple, List
from time import time

import numpy as np
import torch
from fairseq.data import FairseqDataset, LanguagePairDataset, data_utils, Dictionary
from fairseq.tasks.translation import load_langpair_dataset
from torch import Tensor

from .document_config import DocumentConfig
from .document_data_utils import (
    build_segment_index,
    get_tags,
    concat_sents,
    concat_items,
    collate_with_order,
    collate_context,
    filter_index_map_by_size,
    filter_indices_by_size,
)
from .utils import build_mask, word_drop


logger = logging.getLogger(__name__)


def prepare_index(
    data_type: str,
    segment_length: int,
    sentence_num: int,
    docids: np.memmap,
    src_sizes: np.ndarray,
    tgt_sizes: np.ndarray,
    max_sent_length: int,
    sampled: bool = False,
    allow_mixup: bool = False,
    temperure: float = 1.0,  # sampling temperure
    hybrid_sents: List[int] = [1, 2, 4, 8, 999],
    context_num: Tuple[int, int] = (3, 3),
    skip_invalid: bool = False,
    rng: np.random.RandomState = None,
) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    # for sentence level translation, do not build index
    ori_indices = np.arange(len(src_sizes))
    if data_type == "sent2sent":
        if skip_invalid:
            return filter_indices_by_size(
                ori_indices,
                src_sizes,
                tgt_sizes,
                max_sent_length,
            )
        return ori_indices, src_sizes, tgt_sizes

    # prepare index map
    index_map = defaultdict(list)
    for idx, docid in zip(ori_indices, docids):
        index_map[docid].append(idx)
    if skip_invalid:
        index_map = filter_index_map_by_size(
            index_map,
            src_sizes,
            tgt_sizes,
            max_sent_length,
        )

    # build index for document level translation tasks
    if data_type == "divide":
        skipped = 0
        new_indices = []
        new_src_sizes = []
        new_tgt_sizes = []
        for idx, src_size, tgt_size in zip(ori_indices, src_sizes, tgt_sizes):
            if (src_size < 7) or (tgt_size < 7):
                skipped += 1
                continue
            new_indices.append(idx)
            new_src_sizes.append(src_size)
            new_tgt_sizes.append(tgt_size)
        new_src_sizes = np.array(new_src_sizes)
        new_tgt_sizes = np.array(new_tgt_sizes)
        new_indices = np.array(new_indices)
        logger.info(f"{skipped} sentences are skipped.")
    elif data_type == "seg2seg":
        new_indices, new_src_sizes, new_tgt_sizes = build_segment_index(
            index_map=index_map,
            src_sizes=src_sizes,
            tgt_sizes=tgt_sizes,
            max_length=segment_length,
            max_sents=sentence_num,
            sampled=sampled,
            allow_mixup=allow_mixup,
            temperure=temperure,
            rng=rng,
        )
    elif data_type == "hybrid":
        idx_bucket = []
        src_sizes_bucket = []
        tgt_sizes_bucket = []
        for sent_num in hybrid_sents:
            indices, s_sizes, t_sizes = build_segment_index(
                index_map=index_map,
                src_sizes=src_sizes,
                tgt_sizes=tgt_sizes,
                max_length=segment_length,
                max_sents=sent_num,
            )
            idx_bucket.append(indices)
            src_sizes_bucket.append(s_sizes)
            tgt_sizes_bucket.append(t_sizes)
        new_indices = []
        for indices in idx_bucket:
            new_indices.extend(indices)
        new_src_sizes = np.concatenate(src_sizes_bucket)
        new_tgt_sizes = np.concatenate(tgt_sizes_bucket)
    elif data_type == "context":
        prev_indices = []
        future_indices = []
        new_src_sizes, new_tgt_sizes = src_sizes.copy(), tgt_sizes.copy()
        for n, (index, docid) in enumerate(zip(ori_indices, docids)):
            ctx_max, ctx_min = index_map[docid][-1], index_map[docid][0]
            prev_ids = []
            future_ids = []
            for i in range(context_num[0], 0, -1):
                prev_id = index - i
                prev_ids.append(-1 if prev_id < ctx_min else prev_id)
                new_src_sizes[n] += 0 if prev_id < ctx_min else src_sizes[prev_id]
                new_tgt_sizes[n] += 0 if prev_id < ctx_min else tgt_sizes[prev_id]
            for i in range(1, context_num[1] + 1):
                future_id = index + i
                future_ids.append(-1 if future_id > ctx_max else future_id)
                new_src_sizes[n] += 0 if future_id > ctx_max else src_sizes[future_id]
                new_tgt_sizes[n] += 0 if future_id > ctx_max else tgt_sizes[future_id]
            prev_indices.append(prev_ids)
            future_indices.append(future_ids)
        new_indices = ori_indices, prev_indices, future_indices
    elif data_type == "doc2sent":
        src_indices = []
        new_tgt_sizes = tgt_sizes
        new_src_sizes = []
        for index, docid in zip(ori_indices, docids):
            ctx_max = min(index_map[docid][-1], index + context_num[1])
            ctx_min = max(index_map[docid][0], index - context_num[0])
            ctx_indices = np.arange(index - context_num[0], index + context_num[0] + 1)
            ctx_indices[ctx_indices < ctx_min] = -1
            ctx_indices[ctx_indices > ctx_max] = -1
            src_indices.append(ctx_indices)
            new_src_sizes.append(src_sizes[ctx_indices[ctx_indices != -1]].sum())
        new_indices = ori_indices, src_indices
        new_src_sizes = np.array(new_src_sizes)
    else:
        raise TypeError(f"Not supported type: {data_type}")
    return new_indices, new_src_sizes, new_tgt_sizes


class DocumentDataset(FairseqDataset):
    dataset: LanguagePairDataset

    def __init__(
        self,
        data_path: str,
        split: str,
        src: str,
        tgt: str,
        src_dict: Dictionary,
        tgt_dict: Dictionary,
        combine: bool,
        cfg: DocumentConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        # set arguments
        self._epoch = 1
        self.split = split
        self.data_type = cfg.data_type.name
        self.src = src
        self.tgt = tgt
        self.concate_context = cfg.concate_context
        self.sampled = cfg.sampled_segment and (self.data_type == "seg2seg")
        self.adaptive_sample = cfg.adaptive_sample
        self.start_sample_epoch = cfg.start_sample_epoch
        self.spliter = "</s>" if cfg.finetune_mbart else "<eos>"
        self.segment_length = cfg.segment_length
        self.sentence_num = cfg.sentence_num
        seed = getattr(kwargs["model_cfg"], "seed", 2)
        self.seed = seed
        self.data_path = data_path
        self.allow_mixup = cfg.allow_mixup and (split == "train")
        self.max_sent_length = cfg.max_sent_length

        # load langpair dataset
        self.dataset = load_langpair_dataset(
            data_path=data_path,
            split=split,
            src=src,
            tgt=tgt,
            src_dict=src_dict,
            tgt_dict=tgt_dict,
            combine=combine,
            dataset_impl=cfg.dataset_impl,
            upsample_primary=cfg.upsample_primary,
            left_pad_source=cfg.left_pad_source,
            left_pad_target=cfg.left_pad_target,
            max_source_positions=cfg.max_source_positions,
            max_target_positions=cfg.max_target_positions,
            load_alignments=cfg.load_alignments,
            truncate_source=cfg.truncate_source,
            num_buckets=cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=cfg.required_seq_len_multiple,
            append_source_id=(cfg.finetune_mbart == True),
        )

        # prepare index
        start_time = time()
        doc_id = self.load_docid(data_path, split)
        if split == "valid":
            self.sampled = False
            temperure = np.inf
        else:
            temperure = 0.0 if cfg.adaptive_sample else np.inf
        assert len(doc_id) == len(self.dataset)
        logger.info(f"{split} data type: {self.data_type}")
        index, src_sizes, tgt_sizes = prepare_index(
            self.data_type,
            segment_length=cfg.segment_length,
            sentence_num=cfg.sentence_num,
            docids=doc_id,
            src_sizes=self.dataset.src_sizes,
            tgt_sizes=self.dataset.tgt_sizes,
            max_sent_length=self.max_sent_length,
            sampled=self.sampled,
            allow_mixup=self.allow_mixup,
            temperure=temperure,
            hybrid_sents=cfg.hybrid_sents,
            context_num=cfg.context_num,
            skip_invalid=split == "train",
            rng=np.random.RandomState([seed, self._epoch]),
        )
        self.index, self.ctx_index, self.prev_index, self.future_index = (None,) * 4
        if self.data_type == "doc2sent":
            self.index, self.ctx_index = index
            logger.info(f"Previous context num: {cfg.context_num[0]}")
            logger.info(f"Future context num: {cfg.context_num[1]}")
        elif self.data_type == "context":
            self.index, self.prev_index, self.future_index = index
            logger.info(f"Previous context num: {cfg.context_num[0]}")
            logger.info(f"Future context num: {cfg.context_num[1]}")
        else:
            self.index, self.ctx_index = index, None
            logger.info(f"Total segments: {len(index)}")
            logger.info(f"Average segment length: {np.mean(src_sizes):.2f}")
        self.src_sizes = src_sizes
        self.tgt_sizes = tgt_sizes
        end_time = time()
        logger.info(f"Prepare {split} index consuming: {end_time - start_time}")

        # prepare tags/masks
        self.tag_pad_idx = 0  # used for tags
        self.finetune_mbart = cfg.finetune_mbart
        self.src_sp_idx = self.src_dict.eos_index
        self.tgt_sp_idx = self.tgt_dict.eos_index
        self.use_tags = cfg.use_tags
        self.use_mask = cfg.use_mask
        if self.use_mask:
            assert self.use_tags, "'use_tags' is needed to build mask"

        # prepare word_drop
        self._word_drop = cfg.word_drop if split == "train" else 0.0
        self._word_drop_epoch = cfg.word_drop_epoch
        if self._word_drop > 0.0:
            assert cfg.mask_token is not None
            self.src_mask_idx = src_dict.index(cfg.mask_token)
            self.tgt_mask_idx = tgt_dict.index(cfg.mask_token)
        return

    def load_docid(self, data_path: str, split: str) -> np.memmap:
        prefix = os.path.join(data_path, f"{split}.docid")
        if os.path.exists(prefix):
            return np.memmap(prefix, int, "r")
        raise FileNotFoundError(f"Dataset not found: {prefix}")

    @property
    def src_dict(self) -> Dictionary:
        return self.dataset.src_dict

    @property
    def tgt_dict(self) -> Dictionary:
        return self.dataset.tgt_dict

    @property
    def eos(self):
        return self.dataset.eos

    @property
    def src_lang_id(self):
        return self.dataset.src_lang_id

    @property
    def tgt_lang_id(self):
        return self.dataset.tgt_lang_id

    @property
    def left_pad_source(self):
        return self.dataset.left_pad_source

    @property
    def left_pad_target(self):
        return self.dataset.left_pad_target

    @property
    def input_feeding(self):
        return self.dataset.input_feeding

    @property
    def word_drop(self) -> float:
        if self._epoch > self._word_drop_epoch:
            return 0.0
        return self._word_drop

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        if self.sampled:
            return False
        return True

    def set_epoch(self, epoch):
        self._epoch = epoch
        logger.info(f"Word drop ratio: {self.word_drop}")
        if self.sampled and (epoch > 1):
            start_time = time()
            doc_id = self.load_docid(self.data_path, self.split)
            if self.adaptive_sample:
                temperature = np.exp(epoch - self.start_sample_epoch)
            else:
                temperature = np.inf
            self.index, self.src_sizes, self.tgt_sizes = prepare_index(
                self.data_type,
                segment_length=self.segment_length,
                sentence_num=self.sentence_num,
                docids=doc_id,
                src_sizes=self.dataset.src_sizes,
                tgt_sizes=self.dataset.tgt_sizes,
                max_sent_length=self.max_sent_length,
                sampled=True,
                allow_mixup=self.allow_mixup,
                temperure=temperature,
                skip_invalid=self.split == "train",
                rng=np.random.RandomState([self.seed, self._epoch]),
            )
            end_time = time()
            logger.info(f"Resample {self.split} consuming: {end_time - start_time}s")
            logger.info(f"Total segments: {len(self.index)}")
            logger.info(f"Average src segment length: {self.src_sizes.mean():.2f}")
            logger.info(f"Average tgt segment length: {self.tgt_sizes.mean():.2f}")
        return super().set_epoch(epoch)

    def __getitem__(self, index):
        if self.data_type in {"seg2seg", "hybrid"}:
            item = self._getitem1(index)
        elif self.data_type == "doc2sent":
            item = self._getitem2(index)
        elif self.data_type == "context":
            item = self._getitem3(index)
        elif self.data_type in {"sent2sent", "divide"}:
            item = self._getitem4(index)
        else:
            raise TypeError(f"Not supported data_type: {self.data_type}")

        # prepare tags & word_drop
        if self.word_drop > 0:
            item["source"] = word_drop(
                item["source"],
                self.src_mask_idx,
                self.word_drop,
                self.src_dict.nspecial,
                self.src_sp_idx,
            )
            if "target" in item:
                item["prev"] = word_drop(
                    item["target"],
                    self.tgt_mask_idx,
                    self.word_drop,
                    self.tgt_dict.nspecial,
                    self.tgt_sp_idx,
                )
        return item

    def _getitem1(self, index):
        """Used for seg2seg/hybrid data_type

        Args:
            index (int): index
        """
        items = [self.dataset[idx] for idx in self.index[index]]
        # concat items
        item = concat_items(
            id=index,
            items=items,
            src_spliter=self.src_sp_idx,
            tgt_spliter=self.tgt_sp_idx,
            add_tags=self.use_tags,
            eos=self.eos,
            finetune_mbart=self.finetune_mbart,
        )
        del items
        return item

    def _getitem2(self, index):
        """Used for doc2sent data_type

        Args:
            index (int): index
        """

        def fetch_item(index, target=False):
            side = "target" if target else "source"
            if index >= 0:
                return self.dataset[index][side]
            return torch.tensor([self.src_dict.pad_index, self.src_dict.eos_index])

        item = self.dataset[index]
        if self.ctx_index is not None:
            ctx_items = [fetch_item(idx) for idx in self.ctx_index[index]]
            item["source"] = concat_sents(ctx_items, self.src_sp_idx, self.eos)
            if self.use_tags:
                assert self.src_sp_idx is not None, "spliter should be specified"
                item = get_tags(item, self.src_sp_idx, spliter_pos="after")
            del ctx_items
        return item

    def _getitem3(self, index):
        """Used for context data_type

        Args:
            index (int): index
        """

        def fetch_item(index, target=False):
            side = "target" if target else "source"
            if index >= 0:
                return self.dataset[index][side]
            return torch.tensor([self.src_dict.pad_index, self.src_dict.eos_index])

        item = self.dataset[index]
        prev_ids = self.prev_index[index]
        future_ids = self.future_index[index]

        # get source contexts
        if len(prev_ids) > 0:
            item["src_prev"] = [fetch_item(idx) for idx in prev_ids]
            if self.concate_context:
                item["src_prev"] = concat_sents(item["src_prev"])
        if len(future_ids) > 0:
            item["src_future"] = [fetch_item(idx) for idx in future_ids]
            if self.concate_context:
                item["src_future"] = concat_sents(item["src_future"])

        # get target contexts
        if "target" in item:
            if len(prev_ids) > 0:
                item["tgt_prev"] = [fetch_item(idx, True) for idx in prev_ids]
                if self.concate_context:
                    item["tgt_prev"] = concat_sents(item["tgt_prev"])
            if len(future_ids) > 0:
                item["tgt_future"] = [fetch_item(idx, True) for idx in future_ids]
                if self.concate_context:
                    item["tgt_future"] = concat_sents(item["tgt_future"])
        return item

    def _getitem4(self, index):
        """Used for sent2sent data_type

        Args:
            index (int): index
        """

        def divide_context(item, eos):
            split_pos = len(item["source"]) // 2
            src_head = item["source"][:split_pos]
            src_tail = item["source"][split_pos:]
            item["src_prev"] = [torch.cat((src_head, torch.tensor([eos])), dim=0)]
            item["source"] = src_tail
            if "target" in item:
                split_pos = len(item["target"]) // 2
                tgt_head = item["target"][:split_pos]
                tgt_tail = item["target"][split_pos:]
                item["tgt_prev"] = [torch.cat((tgt_head, torch.tensor([eos])), dim=0)]
                item["target"] = tgt_tail
            return item

        item = self.dataset[self.index[index]]
        if self.data_type == "divide":
            item = divide_context(item, self.eos)
        return item

    def filter_indices_by_size(self, indices, max_sizes):
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

    def ordered_indices(self):
        if self.dataset.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def __len__(self):
        return len(self.index)

    @property
    def supports_prefetch(self):
        return False

    def collater(self, samples, with_order=False):
        res, order = collate_with_order(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )
        if res == {}:
            if with_order:
                return res, None
            return res
        src_tokens = res["net_input"]["src_tokens"]
        if self.input_feeding:
            tgt_tokens = res["net_input"]["prev_output_tokens"]

        # collate src tags
        if "src_tag" in samples[0]:
            src_tags = [i["src_tag"] for i in samples]
            src_tags = data_utils.collate_tokens(
                src_tags,
                pad_idx=self.tag_pad_idx,
                eos_idx=self.eos,
                left_pad=self.left_pad_source,
                move_eos_to_beginning=False,
            )
            assert src_tags.shape == src_tokens.shape
            src_tags = src_tags.index_select(0, order)
            res["net_input"]["src_tags"] = src_tags
            if self.use_mask:
                local_mask, global_mask = build_mask(src_tags, src_tags)
                res["net_input"]["src_local_mask"] = local_mask
                res["net_input"]["src_global_mask"] = global_mask

        # collate tgt tags
        if "tgt_tag" in samples[0] and self.input_feeding:
            tgt_tags = [i["tgt_tag"] for i in samples]
            bsz, _ = tgt_tokens.shape
            tgt_tags = data_utils.collate_tokens(
                tgt_tags,
                pad_idx=self.tag_pad_idx,
                eos_idx=self.tag_pad_idx,
                left_pad=self.left_pad_target,
                move_eos_to_beginning=True,
            )
            tgt_tags[:, 0] = 1  # bos(eos) should be with the first sentence
            assert tgt_tags.shape == tgt_tokens.shape
            tgt_tags = tgt_tags.index_select(0, order)
            res["net_input"]["tgt_tags"] = tgt_tags
            if self.use_mask:
                local_mask, global_mask = build_mask(tgt_tags, tgt_tags)
                res["net_input"]["dec_local_mask"] = local_mask
                res["net_input"]["dec_global_mask"] = global_mask
                local_mask, global_mask = build_mask(tgt_tags, src_tags)
                res["net_input"]["crs_local_mask"] = local_mask
                res["net_input"]["crs_global_mask"] = global_mask

        # collate src context
        if "src_prev" in samples[0]:
            res["net_input"]["src_prevs"] = collate_context(
                [i["src_prev"] for i in samples],
                order,
                self.src_dict.pad(),
                self.eos,
                self.left_pad_source,
                isinstance(samples[0]["src_prev"], Tensor),
                False,
            )
        if "src_future" in samples[0]:
            res["net_input"]["src_futures"] = collate_context(
                [i["src_future"] for i in samples],
                order,
                self.src_dict.pad(),
                self.eos,
                self.left_pad_source,
                isinstance(samples[0]["src_future"], Tensor),
                False,
            )

        # collate target context
        if "tgt_prev" in samples[0]:
            res["net_input"]["tgt_prevs"] = collate_context(
                [i["tgt_prev"] for i in samples],
                order,
                self.tgt_dict.pad(),
                self.eos,
                self.left_pad_target,
                isinstance(samples[0]["tgt_prev"], Tensor),
                True,
            )
        if "tgt_future" in samples[0]:
            res["net_input"]["tgt_futures"] = collate_context(
                [i["tgt_future"] for i in samples],
                order,
                self.tgt_dict.pad(),
                self.eos,
                self.left_pad_target,
                isinstance(samples[0]["tgt_future"], Tensor),
                True,
            )
        if with_order:
            return res, order
        return res

    def get_dummy_batch(self, max_lengths: int):
        """Return a dummy batch with a given number of tokens."""
        assert self.data_type == "seg2seg", "only support seg2seg data type"
        return {
            "src_tokens": torch.randint(10, 100, (1, max_lengths)),
            "src_lengths": torch.tensor([max_lengths] * 1),
            "prev_output_tokens": torch.randint(10, 100, (1, max_lengths)),
        }
