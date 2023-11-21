import os
from argparse import Namespace
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import torch
from fairseq.models.transformer import TransformerEncoder
from torch import Tensor

from d2d.utils import build_mask, get_tags

from .length_bias_encoder_layer import LengthBiasEncoderLayer
from .utils import SegmentEmbedding, attn_entropy


class LengthBiasEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        self.layer_counter = 0
        self.head_num = args.encoder.attention_heads
        super().__init__(args, dictionary, embed_tokens, False)
        self.spliter = dictionary.eos_index
        self.need_tags = args.need_tags
        if getattr(args, "seg_emb", False):
            self.seg_emb = SegmentEmbedding(embed_tokens.embedding_dim, 0)
        return

    @property
    def segment_length(self):
        return self.layers[0].segment_length

    def set_seg_length(self, seg_length: float):
        for layer in self.layers:
            if hasattr(layer, "set_seg_length"):
                layer.set_seg_length(seg_length)
        return

    def build_encoder_layer(self, args: Namespace):
        layer = LengthBiasEncoderLayer(args, self.layer_counter)
        self.layer_counter += 1
        return layer

    def forward_embedding(
        self,
        src_tokens,
        token_embedding: Optional[Tensor] = None,
        src_tags: Optional[Tensor] = None,
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if hasattr(self, "seg_emb"):
            assert src_tags is not None, "src_tags is required if seg_emb is enabled"
            x = x + self.seg_emb(src_tags)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # prepare tags and masks
        if self.need_tags:
            tags = self.get_tags(src_tokens, kwargs.get("src_tags", None))
            local_mask, global_mask, ctx_mask = self.build_mask(
                tags,
                kwargs.get("local_mask", None),
                kwargs.get("global_mask", None),
            )
        else:
            tags, local_mask, global_mask, ctx_mask = None, None, None, None

        # embedding
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = encoder_padding_mask.any()
        x, encoder_embedding = self.forward_embedding(
            src_tokens, token_embeddings, tags
        )
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # preserve encoder state
        x = x.transpose(0, 1)
        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        attn = []
        for layer in self.layers:
            x, layer_attn = layer(
                x=x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                tags=tags,
                local_mask=local_mask.clone() if self.need_tags else None,
                global_mask=global_mask.clone() if self.need_tags else None,
                ctx_mask=ctx_mask.clone() if self.need_tags else None,
                need_attn=False,
                sent_embs=kwargs.get("sent_embs", None),
                src_lengths=src_lengths.unsqueeze(-1),
            )
            attn.append(layer_attn)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [src_tokens],
            "src_tags": [tags],
            "src_lengths": [src_lengths],
        }

    def get_tags(self, tokens: Tensor, tags: Optional[Tensor]) -> Tensor:
        """Generate tags for the segment if not provided

        Args:
            tokens (Tensor): token ids [bsz, seq_len]
            tags (Optional[Tensor]): tags [bsz, seq_len]

        Returns:
            Tensor: tags [bsz, seq_len]
        """
        if tags is None:
            tags = get_tags(tokens, self.spliter, self.padding_idx, "after")
        return tags

    @torch.no_grad()
    def build_mask(
        self,
        src_tags: Tensor,
        local_mask: Tensor,
        global_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build local & global mask if not provided.
        Expand the masks if provided.

        Args:
            src_tags (Tensor): tags of the segment [bsz, seq_len]
            local_mask (Tensor): local mask [bsz, seq_len, seq_len]
            global_mask (Tensor): global mask [bsz, seq_len, seq_len]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: returned tensors
        """
        if local_mask is None:
            local_mask, global_mask = build_mask(src_tags, src_tags)
        ctx_mask = (
            (src_tags.max(dim=-1)[0] < 2).unsqueeze(1).expand(-1, src_tags.size(1))
        )
        if ctx_mask.any():
            global_mask.masked_fill_(ctx_mask.unsqueeze(-1), False)
        return local_mask, global_mask, ctx_mask

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        new_out = super().reorder_encoder_out(encoder_out, new_order)
        if encoder_out["src_tags"][0] is not None:
            new_tags = encoder_out["src_tags"][0].index_select(0, new_order)
            new_out["src_tags"] = [new_tags]
        else:
            new_out["src_tags"] = [None]
        return new_out
