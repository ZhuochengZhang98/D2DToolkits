import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import torch
from fairseq.models.transformer import TransformerDecoder
from torch import Tensor
from torch.nn import Embedding

from d2d.utils import build_mask, get_tags

from .length_bias_decoder_layer import LengthBiasDecoderLayer
from .utils import SegmentEmbedding


class LengthBiasDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens: Embedding,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.layer_counter = 0
        self.head_num = args.decoder.attention_heads
        self.spliter = dictionary.eos_index
        super().__init__(args, dictionary, embed_tokens, False, output_projection)
        self.need_tags = args.need_tags
        if getattr(args, "seg_emb", False):
            self.seg_emb = SegmentEmbedding(embed_tokens.embedding_dim, 0)
        return

    @property
    def segment_length(self):
        return self.layers[0].segment_length

    def set_src_length(self, src_length: float):
        for layer in self.layers:
            if hasattr(layer, "set_src_length"):
                layer.set_src_length(src_length)
        return

    def set_tgt_length(self, tgt_length: float):
        for layer in self.layers:
            if hasattr(layer, "set_tgt_length"):
                layer.set_tgt_length(tgt_length)
        return

    def build_decoder_layer(self, args, *largs, **kwargs):
        layer = LengthBiasDecoderLayer(args, self.layer_counter)
        self.layer_counter += 1
        return layer

    def forward_embedding(
        self,
        prev_output_tokens: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        tags: Optional[Tensor] = None,
    ) -> Tensor:
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        prev_output_tokens = prev_output_tokens.contiguous()

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if hasattr(self, "seg_emb"):
            if incremental_state is not None:
                x += self.seg_emb(tags[:, -1:])
            else:
                x += self.seg_emb(tags)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)
        x = x.transpose(0, 1)
        return x

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs,
    ):
        # prepare tags & masks
        src_tags = encoder_out["src_tags"][0]
        tgt_tags = kwargs.get("tgt_tags", None)
        tgt_tags = self.get_tags(tgt_tags, prev_output_tokens)

        dec_local_mask = kwargs.get("dec_local_mask", None)
        dec_global_mask = kwargs.get("dec_global_mask", None)
        crs_local_mask = kwargs.get("crs_local_mask", None)
        crs_global_mask = kwargs.get("crs_global_mask", None)
        (
            dec_local_mask,
            dec_global_mask,
            crs_local_mask,
            crs_global_mask,
            dec_ctx_mask,
            crs_ctx_mask,
        ) = self.build_mask(
            src_tags=src_tags,
            tgt_tags=tgt_tags,
            dec_local_mask=dec_local_mask,
            dec_global_mask=dec_global_mask,
            crs_local_mask=crs_local_mask,
            crs_global_mask=crs_global_mask,
            incremental=incremental_state is not None,
        )

        # forward function
        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        x = self.forward_embedding(prev_output_tokens, incremental_state, tgt_tags)

        self_attn_padding_mask: Optional[Tensor] = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        src_len = encoder_out["src_lengths"][0].unsqueeze(-1)
        tgt_len: Tensor = (
            prev_output_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        if incremental_state is None:
            casual_len = torch.arange(
                1,
                prev_output_tokens.size(1) + 1,
                dtype=prev_output_tokens.dtype,
                device=prev_output_tokens.device,
            ).unsqueeze(0)
            tgt_len = torch.minimum(casual_len, tgt_len)

        # decoder layers
        attn = []
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out=enc,
                encoder_padding_mask=padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
                src_tags=src_tags,
                tgt_tags=tgt_tags,
                dec_local_mask=dec_local_mask.clone() if self.need_tags else None,
                dec_global_mask=dec_global_mask.clone() if self.need_tags else None,
                crs_local_mask=crs_local_mask.clone() if self.need_tags else None,
                crs_global_mask=crs_global_mask.clone() if self.need_tags else None,
                dec_ctx_mask=dec_ctx_mask.clone() if self.need_tags else None,
                crs_ctx_mask=crs_ctx_mask.clone() if self.need_tags else None,
                src_length=src_len,
                tgt_length=tgt_len,
            )
            inner_states.append(x)
            if layer_attn is not None:
                attn.append(layer_attn)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # output projection
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        if not features_only:
            x = self.output_layer(x)
        return x, None

    def get_tags(self, tags: Tensor, prev_output_tokens: Tensor) -> Tensor:
        if not self.need_tags:
            return None
        if tags is None:
            tags = torch.ones_like(prev_output_tokens)
            tags_suffix = get_tags(prev_output_tokens[:, 1:], self.spliter, self.padding_idx)
            tags[:, 1:] = tags_suffix
        return tags

    @torch.no_grad()
    def build_mask(
        self,
        src_tags: Tensor,
        tgt_tags: Tensor,
        dec_local_mask: Tensor = None,
        dec_global_mask: Tensor = None,
        crs_local_mask: Tensor = None,
        crs_global_mask: Tensor = None,
        incremental: bool = False,
    ) -> Tuple[Tensor, ...]:
        """build masks if not provided

        Args:
            src_tags (Tensor): [bsz, src_len]
            tgt_tags (Tensor): [bsz, tgt_len]
            dec_local_mask (Tensor): [bsz, tgt_len, tgt_len]
            dec_global_mask (Tensor): [bsz, tgt_len, tgt_len]
            crs_local_mask (Tensor): [bsz, tgt_len, src_len]
            crs_global_mask (Tensor): [bsz, tgt_len, src_len]

        Returns:
            Tuple[Tensor]:
                dec_local_mask: [bsz, tgt_len, tgt_len]
                dec_global_mask: [bsz, tgt_len, tgt_len]
                crs_local_mask: [bsz, tgt_len, src_len]
                crs_global_mask: [bsz, tgt_len, src_len]
                dec_ctx_mask: [bsz, tgt_len]
                crs_ctx_mask: [bsz, tgt_len]
        """
        if not self.need_tags:
            return (None,) * 6
        dec_tags = tgt_tags[:, -1:] if incremental else tgt_tags
        if dec_local_mask is None:
            dec_local_mask, dec_global_mask = build_mask(dec_tags, tgt_tags)
            crs_local_mask, crs_global_mask = build_mask(dec_tags, src_tags)
        crs_ctx_mask = (
            (dec_tags.max(dim=1)[0] < 2).unsqueeze(1).expand(-1, dec_tags.size(1))
        )
        dec_ctx_mask = (dec_tags == 1) | crs_ctx_mask
        if dec_ctx_mask.any():
            dec_global_mask.masked_fill_(dec_ctx_mask.unsqueeze(-1), False)
        if crs_ctx_mask.any():
            crs_global_mask.masked_fill_(crs_ctx_mask.unsqueeze(-1), False)
        return (
            dec_local_mask,
            dec_global_mask,
            crs_local_mask,
            crs_global_mask,
            dec_ctx_mask,
            crs_ctx_mask,
        )
