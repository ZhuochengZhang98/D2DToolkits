import logging
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor
from fairseq.data import data_utils

from .utils import get_tags


logger = logging.getLogger(__name__)


# TODO: speed optimize
def build_segment_index(
    index_map: Dict[int, List],
    src_sizes: np.ndarray,
    tgt_sizes: np.ndarray,
    max_length: int,
    max_sents: int,
    sampled: bool = False,
    temperure: float = 1.0,  # inf for uniform sampling, 0.0 for sentence-level sampling
    allow_mixup: bool = False,
    rng: np.random.RandomState = None,
) -> Tuple[List, np.ndarray, np.ndarray]:
    # 0 for no limit
    if (max_sents == 0) and (max_length == 0):
        new_indices = [index_map[i] for i in index_map.keys()]
        new_src_sizes = []
        new_tgt_sizes = []
        for indices in new_indices:
            new_src_sizes.append(src_sizes[indices].sum())
            new_tgt_sizes.append(tgt_sizes[indices].sum())
        new_src_sizes = np.array(new_src_sizes)
        new_tgt_sizes = np.array(new_tgt_sizes)
        return new_indices, new_src_sizes, new_tgt_sizes

    new_indices = []
    new_src_sizes = []
    new_tgt_sizes = []
    seg_len = max_length
    seq_num = max_sents if max_sents > 0 else np.inf
    if sampled:
        if temperure == 0:
            tau = np.inf
        else:
            tau = 1 / temperure
        sample_range = np.arange(max_length)
        init_weight = np.array(1 / np.exp(np.arange(max_length)))
        sample_weight = init_weight**tau / np.sum(init_weight**tau)
        logger.info(f"Segment lengths are sampled with temperature={temperure}")

    # build segment indices
    indices = []
    src_len, tgt_len = 0, 0
    if sampled:
        seg_len = rng.choice(sample_range, p=sample_weight)
    for docid, sents in index_map.items():
        for sent_id in sents:
            # reach segment limit
            if (
                (src_len + src_sizes[sent_id] >= seg_len)
                or (tgt_len + tgt_sizes[sent_id] >= seg_len)
                or (len(indices) >= seq_num)
            ) and (len(indices) > 0):
                new_indices.append(indices)
                new_src_sizes.append(src_len)
                new_tgt_sizes.append(tgt_len)
                if sampled:  # sample new segment length
                    seg_len = rng.choice(sample_range, p=sample_weight)
                indices = []
                src_len, tgt_len = 0, 0
                if sampled:
                    seg_len = rng.choice(sample_range, p=sample_weight)
            # append new sentence
            src_len += src_sizes[sent_id]
            tgt_len += tgt_sizes[sent_id]
            indices.append(sent_id)
        # reach doc end
        if (len(indices) > 0) and (not allow_mixup):
            new_indices.append(indices)
            new_src_sizes.append(src_len)
            new_tgt_sizes.append(tgt_len)
            indices = []
            src_len, tgt_len = 0, 0
            if sampled:
                seg_len = rng.choice(sample_range, p=sample_weight)
    # remaining segment
    if len(indices) > 0:
        new_indices.append(indices)
        new_src_sizes.append(src_len)
        new_tgt_sizes.append(tgt_len)
    new_src_sizes = np.array(new_src_sizes)
    new_tgt_sizes = np.array(new_tgt_sizes)
    return new_indices, new_src_sizes, new_tgt_sizes


def get_tags_full(
    sample: Dict[str, Tensor],
    src_spliter: int,
    tgt_spliter: int = None,
) -> Dict[str, Tensor]:
    sample["src_tag"] = get_tags(sample["source"], src_spliter, spliter_pos="after")
    if ("target" in sample) and (tgt_spliter is not None):
        sample["tgt_tag"] = get_tags(
            sample["target"], tgt_spliter, spliter_pos="before"
        )
    return sample


def concat_sents(
    sents: List[Tensor],
    spliter: Optional[int] = None,
    eos: int = None,
    finetune_mbart: bool = False,
) -> Tensor:
    """Concate list of token_ids into one tensor.

    Args:
        sents (List[Tensor]): List of token_ids
        spliter (Optional[int], optional): spliter token. Defaults to None.
        eos (int, optional): eos token. Defaults to None.

    Returns:
        Tensor
    """
    if len(sents) == 0:
        return sents
    if spliter is None:
        out = [i[:-1] for i in sents[:-1]]
        out.append(sents[-1])
    elif finetune_mbart:
        out = [i[:-1] for i in sents[:-1]]
        out.append(torch.cat((sents[-1][:-2], sents[-1][-1:])))  # remove last eos
    else:
        out = [i for i in sents]
    out = torch.cat(out)
    if (spliter is not None) and (not finetune_mbart):
        mask = out == eos
        mask[-1] = False
        out.masked_fill_(mask, spliter)
    return out


def split_docs(docs: Tensor, spliter: int, padding_idx: int = None) -> List[Tensor]:
    if padding_idx is not None:
        docs = docs[docs != padding_idx]
    indices = torch.nonzero(docs == spliter).squeeze(-1)
    indices = torch.cat([torch.tensor([-1], device=indices.device), indices])
    sizes = indices[1:] - indices[:-1]
    sents = torch.split_with_sizes(docs, sizes.tolist())
    return sents


def concat_items(
    id: int,
    items: List[Dict[str, Tensor]],
    src_spliter: Optional[int] = None,
    tgt_spliter: Optional[int] = None,
    add_tags: bool = False,
    eos: int = None,
    finetune_mbart: bool = False,
) -> Dict[str, Tensor]:
    if len(items) == 0:
        return None
    else:
        sample = {"id": id}
        srcs = [i["source"] for i in items]
        sample["source"] = concat_sents(srcs, src_spliter, eos, finetune_mbart)
        if "target" in items[0]:
            tgts = [i["target"] for i in items]
            sample["target"] = concat_sents(tgts, tgt_spliter, eos, finetune_mbart)
    if add_tags:
        sample = get_tags_full(sample, src_spliter, tgt_spliter)
    return sample


def collate_with_order(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
) -> Tuple[dict, Optional[Tensor]]:
    if len(samples) == 0:
        return {}, None

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            prev_key = "prev" if "prev" in samples[0] else "target"
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                prev_key,
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints.index_select(0, sort_order)

    return batch, sort_order


def collate_context(
    samples: List[Tensor],
    order: Tensor,
    pad_idx: int,
    eos_idx: int,
    left_pad: bool,
    concate_context: bool,
    move_eos_to_beginning: bool,
):
    if concate_context:
        context = data_utils.collate_tokens(
            values=[i for i in samples],
            pad_idx=pad_idx,
            eos_idx=eos_idx,
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
        )
        return context.index_select(0, order)

    contexts = []
    for n in range(len(samples[0])):
        context = data_utils.collate_tokens(
            values=[i[n] for i in samples],
            pad_idx=pad_idx,
            eos_idx=eos_idx,
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
        )
        contexts.append(context.index_select(0, order))
    return contexts


def filter_index_map_by_size(
    index_map: Dict[int, List[int]],
    src_sizes: np.ndarray,
    tgt_sizes: np.ndarray,
    max_sent_len: int,
) -> Dict[int, List[int]]:
    ignored = []
    for doc_id, sent_indices in index_map.items():
        if src_sizes[sent_indices].max() > max_sent_len:
            ignored.append(doc_id)
            continue
        if tgt_sizes[sent_indices].max() > max_sent_len:
            ignored.append(doc_id)
            continue
    for doc_id in ignored:
        index_map.pop(doc_id)
    logger.warning(
        f"{len(ignored)} documents have invalid sentence length and will be filtered."
    )
    return index_map


def filter_indices_by_size(
    indices: np.ndarray,
    src_sizes: np.ndarray,
    tgt_sizes: np.ndarray,
    max_sent_len: int,
) -> Dict[int, List[int]]:
    ignored_mask = src_sizes > max_sent_len
    ignored_mask |= tgt_sizes > max_sent_len
    indices = indices[~ignored_mask]
    src_sizes = src_sizes[~ignored_mask]
    tgt_sizes = tgt_sizes[~ignored_mask]
    logger.warning(
        f"{ignored_mask.sum()} sentences have invalid length and will be filtered."
    )
    return indices, src_sizes, tgt_sizes
