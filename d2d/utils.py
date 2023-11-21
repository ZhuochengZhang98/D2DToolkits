import traceback
from typing import Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from fairseq.data import Dictionary
from fairseq.modules import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding


@torch.no_grad()
def build_mask(
    query_tags: Tensor,
    key_tags: Tensor,
    incremental: bool = False,
    head_num: int = None,
) -> Tuple[Tensor, Tensor]:
    if incremental:
        query_tags = query_tags[:, -1:]
    local_mask: Tensor = key_tags.unsqueeze(1) != query_tags.unsqueeze(2)
    global_mask: Tensor = ~local_mask
    local_mask &= 0 != query_tags.unsqueeze(2)
    global_mask &= 0 != query_tags.unsqueeze(2)
    if head_num is not None:
        local_mask = (
            local_mask.unsqueeze(1).expand(-1, head_num, -1, -1).flatten(0, 1)
        )  # [bsz * heads, seq_len, seq_len]
        global_mask = (
            global_mask.unsqueeze(1).expand(-1, head_num, -1, -1).flatten(0, 1)
        )  # [bsz * heads, seq_len, seq_len]
    return local_mask, global_mask


@torch.no_grad()
def get_tags(
    tokens: Tensor,
    spliter: int,
    padding_idx: int = None,
    spliter_pos: str = "before",
) -> Tensor:
    tags = torch.cumsum(tokens == spliter, dim=-1) + 1
    if spliter_pos == "after":
        tags[tokens == spliter] -= 1
    elif spliter_pos == "before":
        pass
    else:
        raise TypeError(f"Unknown spliter_pos: {spliter_pos}")
    if padding_idx is not None:
        padding_mask = tokens == padding_idx
        tags.masked_fill_(padding_mask, 0.0)
    return tags


@torch.no_grad()
def word_drop(
    tokens: Tensor,
    mask_idx: int,
    drop_ratio: float,
    nspecial: int,
    sp_idx: int,
) -> Tensor:
    mask = torch.rand_like(tokens, dtype=torch.float) < drop_ratio
    mask &= (tokens >= nspecial) & (tokens != sp_idx)  # prevent drop special tokens
    # mask &= tokens < len(dict) - self.naddition
    tokens = torch.where(mask, tokens.new_full(tokens.size(), mask_idx), tokens)
    return tokens


def is_ampere() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        device_name = torch.cuda.get_device_name(0)
    except:
        return False
    if "A100" in device_name:
        return True
    if "A40" in device_name:
        return True
    return False


class FinetunedDictionary(Dictionary):
    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator=" ",
    ):
        text = super().string(
            tensor,
            bpe_symbol,
            escape_unk,
            extra_symbols_to_ignore,
            unk_string,
            True,
            separator,
        )
        return text


class SegmentEmbedding(LearnedPositionalEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, positions: Tensor):
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class SegmentEmbeddingSin(SinusoidalPositionalEmbedding):
    def __init__(self, embedding_dim: int, padding_idx: int = 0, init_size: int = 1024):
        super().__init__(embedding_dim, padding_idx, init_size)

    def forward(self, positions: Tensor):
        bsz, seq_len = positions.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )


def on_generate():
    stacks = traceback.format_stack()
    for stack in stacks:
        if "fairseq_cli/generate.py" in stack:
            return True
    return False
