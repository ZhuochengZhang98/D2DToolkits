import math
from typing import Optional

from fairseq.modules import TransformerEncoderLayer, LayerNorm
from torch import Tensor

from .utils import AttnGate
from .weighted_attention import WeightedAttention
from .legacy_utils import make_attn_mask, PairwiseDistance


class LengthBiasEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, counter: int):
        self.layer_counter = counter
        self.layer_type = args.enc_type[counter]
        super().__init__(args)

        if self.layer_type in ("hybrid", "group"):
            self.global_attn = self.build_self_attention(
                self.embed_dim, args, "enc" in args.length_norm
            )
            self.attn_gate = self.build_attn_gate(self.embed_dim)
        elif self.layer_type == "stack":
            self.global_attn = self.build_self_attention(
                self.embed_dim, args, "enc" in args.length_norm
            )
            self.global_attn_norm = LayerNorm(self.embed_dim, export=args.export)
        elif self.layer_type == "global":  # rebuild attn with length norm
            self.self_attn = self.build_self_attention(self.embed_dim, args)

        if args.freeze:
            self.self_attn.requires_grad_(False)
            self.self_attn_layer_norm.requires_grad_(False)
            self.fc1.requires_grad_(False)
            self.fc2.requires_grad_(False)
            self.final_layer_norm.requires_grad_(False)
        return

    @property
    def segment_length(self):
        src_len = getattr(self.self_attn, "length_norm", None)
        if src_len is not None:
            src_len = math.exp(float(src_len))
        return src_len

    def set_seg_length(self, seg_length: float):
        if hasattr(self, "self_attn"):
            if hasattr(self.self_attn, "reset_length_norm"):
                self.self_attn.reset_length_norm(seg_length)
        if hasattr(self, "global_attn"):
            if hasattr(self.global_attn, "reset_length_norm"):
                self.global_attn.reset_length_norm(seg_length)
        return

    def build_attn_gate(self, embed_dim: int):
        return AttnGate(embed_dim)

    def build_self_attention(self, embed_dim, args, length_norm: bool = False):
        if ("enc" in args.length_norm) and (self.layer_type == "global"):
            average_length = 1.0
        elif length_norm:
            average_length = 1.0
        else:
            average_length = None
        return WeightedAttention(
            embed_dim,
            args.encoder.attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            average_length=average_length,
        )

    def forward(
        self,
        x,
        tags: Tensor,
        local_mask: Tensor,
        global_mask: Tensor,
        ctx_mask: Tensor,
        encoder_padding_mask: Optional[Tensor],
        **kwargs,
    ):
        # update masks
        if self.layer_type == "global":
            local_mask = None
        elif self.layer_type == "group":
            global_mask = None
            ctx_mask = None

        attn = {}

        # local attention
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if hasattr(self.self_attn, "length_norm"):
            addition_args = {"kv_length": kwargs.get("src_lengths", None)}
        else:
            addition_args = {}
        x_local, attn_weight = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=kwargs.get("need_attn", False),
            attn_mask=local_mask,
            **addition_args,
        )
        attn["enc"] = attn_weight

        # global attention
        if (self.layer_type == "hybrid") or (self.layer_type == "group"):
            x_global, attn_weight = self.global_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=kwargs.get("need_attn", False),
                attn_mask=global_mask,
                kv_length=kwargs.get("src_lengths", None),
            )
            x = self.attn_gate(x_local, x_global, ctx_mask)
            attn[f"enc_{self.layer_type}"] = attn_weight
        elif self.layer_type == "stack":
            x_local = self.dropout_module(x_local)
            x_local = self.residual_connection(x_local, residual)
            if not self.normalize_before:
                x_local = self.self_attn_layer_norm(x_local)
            else:
                x_local = self.global_attn_norm(x_local)
            residual = x_local
            x_global, _ = self.global_attn(
                query=x_local,
                key=x_local,
                value=x_local,
                key_padding_mask=encoder_padding_mask,
                need_weights=kwargs.get("need_attn", False),
                attn_mask=global_mask,
                kv_length=kwargs.get("src_lengths", None),
            )
            x = x_global.masked_fill(ctx_mask.transpose(0, 1).unsqueeze(-1), 0.0)
            attn[f"enc_{self.layer_type}"] = attn_weight
        elif self.layer_type == "selective":
            x_global, _ = self.global_attn(
                query=x,
                key=x,
                value=x,
                key_tags=tags,
                query_tags=tags,
                key_padding_mask=encoder_padding_mask,
                query_padding_mask=encoder_padding_mask,
                attn_mask=global_mask,
            )
            x = self.attn_gate(x_local, x_global, ctx_mask)
        elif self.layer_type == "legacy":
            assert kwargs["sent_embs"] is not None
            global_mask, ctx_mask = make_attn_mask(
                sent_embs=kwargs["sent_embs"],
                query_tags=tags,
                key_tags=tags,
                metric=self.metric,
                origin_mask=global_mask,
                ctx_mask=ctx_mask,
                attn_type="enc",
                order=2,
            )
            x_global, _ = self.global_attn(
                query=x,
                key=x,
                value=x,
                key_tags=tags,
                query_tags=tags,
                key_padding_mask=encoder_padding_mask,
                query_padding_mask=encoder_padding_mask,
                attn_mask=global_mask,
            )
            x = self.attn_gate(x_local, x_global, ctx_mask)
        else:
            x = x_local

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            if hasattr(self, "global_attn_norm"):
                x = self.global_attn_norm(x)
            else:
                x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn
