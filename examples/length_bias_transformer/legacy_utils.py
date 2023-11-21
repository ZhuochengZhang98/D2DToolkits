from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PairwiseCosine(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        eps = 2e-4 if x1.dtype == torch.float16 else 1e-8
        if x1.dim() == 2:
            x1_mat = x1.unsqueeze(0).expand(x2.size(0), -1, -1)
            x2_mat = x2.unsqueeze(1).expand(-1, x1.size(0), -1)
            pairwise_cos = F.cosine_similarity(x1_mat, x2_mat, dim=2, eps=eps)
        elif x1.dim() == 3:
            x1_mat = x1.unsqueeze(1).expand(-1, x2.size(1), -1, -1)
            x2_mat = x2.unsqueeze(2).expand(-1, -1, x1.size(1), -1)
            pairwise_cos = F.cosine_similarity(x1_mat, x2_mat, dim=3, eps=eps)
        else:
            raise ValueError("Unexpected dimension")
        return pairwise_cos


class PairwisePNorm(nn.Module):
    def __init__(self, p=1.0) -> None:
        super().__init__()
        self.metric = nn.PairwiseDistance(p=p)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        assert x1.dim() == x2.dim()
        if x1.dim() == 2:
            x1_mat = x1.unsqueeze(0).expand(x2.size(0), -1, -1)
            x2_mat = x2.unsqueeze(1).expand(-1, x1.size(0), -1)
            pw = self.metric(x1_mat, x2_mat)
        elif x1.dim() == 3:
            x1_mat = x1.unsqueeze(1).expand(-1, x2.size(1), -1, -1)
            x2_mat = x2.unsqueeze(2).expand(-1, -1, x1.size(1), -1)
            pw = self.metric(x1_mat, x2_mat)
        else:
            raise ValueError("Unexpected dimension")
        return pw


class PairwiseDistance(nn.Module):
    def __init__(self, metric: str = "cos") -> None:
        super().__init__()
        self.metric_type = metric
        if metric == "l1":
            self.metric = PairwisePNorm(1)
        elif metric == "l2":
            self.metric = PairwisePNorm(2)
        elif metric == "cos":
            self.metric = PairwiseCosine()
        else:
            raise TypeError(f"Not supported metric type {metric}")

    def forward(self, x: Tensor):
        return self.metric(x, x)

    @property
    def min_value(self):
        if self.metric_type == "cos":
            return -1.0
        else:
            return 0.0

    @property
    def max_value(self):
        if self.metric_type == "cos":
            return 1.0
        else:
            return torch.inf


def make_attn_mask(
    sent_embs: Tensor,
    query_tags: Tensor,
    key_tags: Tensor,
    metric: PairwiseDistance,
    origin_mask: Tensor,
    ctx_mask: Tensor,
    attn_type: str,
    order: int,
) -> Tuple[Tensor]:
    # build similarity mat
    sent_sims = metric(sent_embs)
    sent_sims.diagonal(dim1=1, dim2=2).fill_(metric.min_value)
    sent_idx = torch.arange(0, sent_embs.size(1), device=query_tags.device).unsqueeze(0)
    sent_pad = sent_idx > query_tags.max(dim=1, keepdim=True)[0]
    sent_pad = sent_pad.unsqueeze(1) | sent_pad.unsqueeze(2)
    sent_sims.masked_fill_(sent_pad, metric.min_value)
    if attn_type == "dec":
        raise NotImplementedError
    elif attn_type == "crs":
        raise NotImplementedError

    # calc threshold
    threshold = sent_sims.topk(dim=-1, k=order)[0][:, :, order - 1]
    threshold = threshold - 1e-4
    threshold = torch.gather(threshold, index=query_tags, dim=-1)

    # build token level mask
    sents = sent_embs.size(1)
    toks = query_tags.size(1)
    tok_sims = sent_sims.gather(1, query_tags.unsqueeze(-1).expand(-1, -1, sents))
    tok_sims = tok_sims.gather(2, key_tags.unsqueeze(1).expand(-1, toks, -1))
    slt_mask = tok_sims < threshold.unsqueeze(-1)


    # post-process
    slt_mask = slt_mask | origin_mask
    slt_mask = slt_mask.masked_fill((query_tags == 0).unsqueeze(-1), False)
    return slt_mask, ctx_mask
