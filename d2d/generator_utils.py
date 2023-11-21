from typing import Dict, List, Tuple
from collections import defaultdict
from itertools import zip_longest

import torch
from torch import Tensor
from fairseq.data.data_utils import collate_tokens

from .document_data_utils import get_tags


def build_prefix(
    hypothesis: List[List[Dict[str, Tensor]]],
    prefix_num: List[int],
    eos_id: int = 2,
    pad_id: int = 1,
) -> Tensor:
    """Build prefix for slide window decoding

    Args:
        hypothesis (List[List[Dict[str, Tensor]]]): generated hypothesis
        prefix_num (List[int]): number of sentences for prefix

    Returns:
        Tensor: prefix tokens
    """
    prefixes = []
    for hypo, num in zip(hypothesis, prefix_num):
        if num == 0:
            continue
        hypo = hypo[0]["tokens"]
        hypo_tags = get_tags(hypo, spliter=eos_id, spliter_pos="after")
        hypo_mask = hypo_tags > (hypo_tags[-1] - num)
        prefixes.append(hypo[hypo_mask])
    if prefixes == []:
        return None
    return collate_tokens(prefixes, pad_idx=pad_id, left_pad=False)


def construct_final(
    hypothesis: List[List[List[Dict[str, Tensor]]]],
    final_num_batches: List[List[int]],
    new_batch_order: List[int],
    eos_id: int = 2,
) -> List[List[Dict[str, Tensor]]]:
    """Construct final output

    Args:
        hypothesis (List[List[List[Dict[str, Tensor]]]]): generated hypothesis of each batch
        final_num_batches (List[List[int]]): the generated sentence number of each batch
        new_batch_order (List[int]): the order of the batch

    Returns:
        List[List[Dict[str, Tensor]]]: concatenated hypothesis
    """
    beam = len(hypothesis[0][0])
    bsz = len(hypothesis[0])
    finalized = [[defaultdict(list) for _ in range(beam)] for _ in range(bsz)]

    for final_nums, hypo_batch in zip(final_num_batches, hypothesis):
        for batch_id, (final_num, hypo) in enumerate(zip(final_nums, hypo_batch)):
            for beam_id, hypo_beam in enumerate(hypo):
                hypo_tags = get_tags(hypo_beam["tokens"], eos_id, spliter_pos="after")
                hypo_mask = hypo_tags > (hypo_tags[-1] - final_num)
                hypo_token = hypo_beam["tokens"][hypo_mask]
                hypo_score = hypo_beam["positional_scores"][hypo_mask]
                finalized[batch_id][beam_id]["tokens"].append(hypo_token)
                finalized[batch_id][beam_id]["positional_scores"].append(hypo_score)

    for batch_id, hypo_batch in enumerate(finalized):
        for beam_id, hypo_beam in enumerate(hypo_batch):
            hypo_beam["tokens"] = torch.cat(hypo_beam["tokens"])
            hypo_beam["positional_scores"] = torch.cat(hypo_beam["positional_scores"])
            hypo_beam["score"] = hypo_beam["positional_scores"].sum()
            hypo_beam["alignment"] = torch.tensor([])
            hypo_beam["attention"] = torch.tensor([])

    # recover batch order
    order = sorted(range(len(new_batch_order)), key=lambda i: new_batch_order[i])
    finalized = [finalized[i] for i in order]
    return finalized


def build_context_window_item(
    src_tokens: Tensor,
    context_window: int,
    eos_id: int = 2,
    pad_id: int = 1,
) -> Tuple[List[Tensor], List[int], List[Tuple[int, int]]]:
    """Construct context window for a single document

    Args:
        src_tokens (Tensor): source tokens
        context_window (int): context window size
        eos_id (int, optional): EOS token id. Defaults to 2.
        pad_id (int, optional): PAD token id. Defaults to 1.

    Returns:
        slide source tokens (List[Tensor])
        slide source lengths (List[int])
        slide window positions (List[Tuple[int, int]])
    """
    src_tokens = src_tokens[src_tokens != pad_id]
    src_tags = get_tags(src_tokens, eos_id, spliter_pos="after")
    if context_window >= src_tokens.size(0):
        return [src_tokens], [src_tokens.size(0)], [(1, 1 + src_tags[-1])]

    src_tags = get_tags(src_tokens, eos_id, spliter_pos="after")

    new_src_sents = []
    new_src_lengths = []
    window_pos = []

    start_point = 1
    end_point = src_tags[context_window].item()
    src_mask = (src_tags >= start_point) & (src_tags < end_point)
    while end_point <= (src_tags[-1] + 1):
        window_pos.append((start_point, end_point))
        new_src_sents.append(src_tokens[src_mask])
        new_src_lengths.append(src_mask.sum().item())
        # move at least 1 step forward
        end_point += 1
        src_mask = (src_tags >= start_point) & (src_tags < end_point)
        while src_mask.sum() > context_window:
            start_point += 1
            src_mask = (src_tags >= start_point) & (src_tags < end_point)
        while (end_point <= src_tags[-1]) and (
            src_mask.sum() + (src_tags == end_point).sum() <= context_window
        ):
            end_point += 1
            src_mask = (src_tags >= start_point) & (src_tags < end_point)
    return new_src_sents, new_src_lengths, window_pos


def build_context_window(
    sample: Dict[str, Dict[str, Tensor]],
    context_window: int,
    eos_id: int,
    pad_id: int,
) -> Tuple[List[Tensor], List[List[int]], List[List[int]], List[int]]:
    """split the document to maximize the context window

    Args:
        sample (Dict[str, Dict[str, Tensor]]): samples from dataset
        context_window (int): context window size
        eos_id (int): eos token id
        pad_id (int): pad token id

    Returns:
        src_token_batches (List[Tensor]): slide source tokens
        src_lengths_batches (List[List[int]]): slide source lengths
        window_pos_batches (List[List[int]]): slide window positions
        new_batch_order (List[int]): the order of the new batch
    """
    new_src_sents = []
    new_src_lengths = []
    window_poss = []
    for src_item in sample["net_input"]["src_tokens"]:
        src_sents, src_lengths, window_pos = build_context_window_item(
            src_item,
            context_window,
            eos_id,
            pad_id,
        )
        new_src_sents.append(src_sents)
        new_src_lengths.append(src_lengths)
        window_poss.append(window_pos)
    slide_num = [len(item) for item in new_src_sents]
    new_batch_order = sorted(
        range(len(slide_num)),
        key=lambda i: slide_num[i],
        reverse=True,
    )

    # reorder the splitted documents
    new_src_sents = [new_src_sents[i] for i in new_batch_order]
    new_src_lengths = [new_src_lengths[i] for i in new_batch_order]
    window_poss = [window_poss[i] for i in new_batch_order]

    # rebatch the splitted documents
    src_token_batches = []
    src_lengths_batches = []
    window_pos_batches = []
    # prefix_num_batches = []
    for src_items in zip_longest(*new_src_sents):
        src_items = [item for item in src_items if item is not None]
        src_token_batches.append(
            collate_tokens(src_items, pad_idx=pad_id, left_pad=True)
        )
    for src_lengths in zip_longest(*new_src_lengths):
        src_lengths = [item for item in src_lengths if item is not None]
        src_lengths_batches.append(torch.tensor(src_lengths).to(src_token_batches[0]))
    for window_pos in zip_longest(*window_poss):
        window_pos = [item for item in window_pos if item is not None]
        window_pos_batches.append(window_pos)

    return (
        src_token_batches,
        src_lengths_batches,
        window_pos_batches,
        new_batch_order,
    )
