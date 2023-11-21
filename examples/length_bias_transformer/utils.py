import os
from typing import List, Optional, Dict, NamedTuple, Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from fairseq.modules import SinusoidalPositionalEmbedding


class MLP(nn.Module):
    def __init__(self, dims: List[int], dropout, bias=True, act="relu") -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        self.dropout = dropout
        self.layer_num = len(layers)
        self.layers = nn.Sequential(*layers)
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        if act == "softmax":
            self.act = nn.Softmax(2)
        if act == "tanh":
            self.act = nn.Tanh()
        if act == "relu":
            self.act = nn.ReLU()

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(x)
        return x

    def reset_parameters(self):
        for i in range(self.layer_num):
            nn.init.xavier_uniform_(self.layers[i].weight)
            if self.layers[i].bias is not None:
                nn.init.constant_(self.layers[i].bias, 0.0)


class AttnGate(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ffn = torch.nn.Linear(embed_dim * 2, embed_dim)
        self.act = nn.Sigmoid()
        return

    def forward(
        self, x_local: Tensor, x_global: Tensor, ctx_mask: Optional[Tensor] = None
    ) -> Tensor:
        """forward function for AttnGate

        Args:
            x_local (Tensor): [seq, bsz, emb]
            x_global (Tensor): [seq, bsz, emb]
            ctx_mask (Tensor): [bsz, seq]

        Returns:
            Tensor: [seq, bsz, emb]
        """
        # calc gate
        g = self.ffn(torch.cat([x_local, x_global], dim=-1))
        g = self.act(g)
        # mask the contextless sentences
        if ctx_mask is not None:
            mask = ctx_mask.transpose(0, 1).unsqueeze(-1)  # [seq, bsz, 1]
            g = torch.masked_fill(g, mask, 1.0)
        # perform gate
        x = x_local * g + x_global * (1 - g)
        return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def attn_entropy(attn, reduce=False) -> np.ndarray:
    with torch.no_grad():
        attn = attn.detach().to(torch.float32).clamp(1e-8, 1.0)
        entropy = -(attn * torch.log(attn)).sum(dim=-1)
        if reduce:
            entropy = entropy.mean(dim=1)
        return entropy.cpu().numpy()


def get_metric(metric: str = "cos") -> nn.Module:
    if metric == "l1":
        return nn.PairwiseDistance(p=1)
    elif metric == "l2":
        return nn.PairwiseDistance(p=2)
    elif metric == "cos":
        return PairwiseCosine()
    else:
        raise TypeError(f"Not supported metric type {metric}")


def gumbel_topk(logits: Tensor, tau: int, k: int, dim: int = -1):
    for _ in range(k):
        y = F.gumbel_softmax(logits, tau, dim=dim)
    return y


class PairwiseDistance(nn.Module):
    def __init__(self, metric: str = "cos") -> None:
        super().__init__()
        if metric == "l1":
            self.metric = PairwisePNorm(1)
        elif metric == "l2":
            self.metric = PairwisePNorm(2)
        elif metric == "cos":
            self.metric = PairwiseCosine()
        else:
            raise TypeError(f"Not supported metric type {metric}")

    def forward(self, x1: Tensor, x2: Tensor):
        return self.metric(x1, x2)


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


EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_tags", Tensor),  # B x T
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("encoder_attn", Optional[Dict[str, Tensor]]),  # Dict[B x T]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
        ("sim_mats", Optional[List[Tensor]]),  # List[sents, sents]
        ("uid", Optional[str]),  # uuid
    ],
)


def exist_or_make(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return


class SegmentEmbedding(SinusoidalPositionalEmbedding):
    def forward(self, positions: Optional[Any] = None):
        bsz, seq_len = positions.shape
        max_pos = positions.max()
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
