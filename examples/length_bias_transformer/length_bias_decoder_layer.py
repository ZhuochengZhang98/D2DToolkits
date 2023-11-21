import math
from argparse import Namespace
from typing import Dict, Optional

from fairseq.modules import TransformerDecoderLayer, LayerNorm
from torch import Tensor

from .utils import AttnGate
from .weighted_attention import WeightedAttention


class LengthBiasDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, layer_counter: int):
        """Length Bias Decoder Layer
        Args:
            args (_type_): _description_
            layer_counter (int): _description_

        Raises:
            NotImplementedError: _description_
        """
        self.layer_counter = layer_counter
        self.dec_type = args.dec_type[layer_counter]
        self.crs_type = args.crs_type[layer_counter]
        super().__init__(args, False, False, False)

        # build global layers
        if self.dec_type in ("hybrid", "group"):
            self.dec_global_attn = self.build_self_attention(
                self.embed_dim, args, length_norm="dec" in args.length_norm
            )
            self.dec_attn_gate = self.build_attn_gate(self.embed_dim)
        elif self.dec_type == "stack":
            self.dec_global_attn = self.build_self_attention(
                self.embed_dim, args, length_norm="dec" in args.length_norm
            )
            self.dec_global_norm = LayerNorm(self.embed_dim, export=args.export)
        elif self.dec_type == "global":  # rebuild attn with length norm
            self.self_attn = self.build_self_attention(self.embed_dim, args)

        if self.crs_type in ("hybrid", "group"):
            self.crs_global_attn = self.build_encoder_attention(
                self.embed_dim, args, length_norm="crs" in args.length_norm
            )
            self.crs_attn_gate = self.build_attn_gate(self.embed_dim)
        elif self.crs_type == "stack":
            self.crs_global_attn = self.build_encoder_attention(
                self.embed_dim, args, length_norm="crs" in args.length_norm
            )
            self.crs_global_norm = LayerNorm(self.embed_dim, export=args.export)
        elif self.crs_type == "global":  # rebuild attn with length norm
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)

        # Not supported arguments
        assert self.c_attn is None
        assert not self.cross_self_attention
        assert not self.onnx_trace
        assert self.encoder_attn is not None
        assert self.w_resid is None

        # freeze
        if args.freeze:
            self.encoder_attn.requires_grad_(False)
            self.encoder_attn_layer_norm.requires_grad_(False)
            self.self_attn.requires_grad_(False)
            self.self_attn_layer_norm.requires_grad_(False)
            self.fc1.requires_grad_(False)
            self.fc2.requires_grad_(False)
            self.final_layer_norm.requires_grad_(False)
        return

    @property
    def segment_length(self):
        src_len = getattr(self.encoder_attn, "length_norm", None)
        tgt_len = getattr(self.self_attn, "length_norm", None)
        if src_len is not None:
            src_len = math.exp(float(src_len))
        if tgt_len is not None:
            tgt_len = math.exp(float(tgt_len))
        return src_len, tgt_len

    def set_src_length(self, src_length: float):
        if hasattr(self, "encoder_attn"):
            if hasattr(self.encoder_attn, "reset_length_norm"):
                self.encoder_attn.reset_length_norm(src_length)
        if hasattr(self, "crs_global_attn"):
            if hasattr(self.crs_global_attn, "reset_length_norm"):
                self.crs_global_attn.reset_length_norm(src_length)
        return

    def set_tgt_length(self, tgt_length: float):
        if hasattr(self, "self_attn"):
            if hasattr(self.self_attn, "reset_length_norm"):
                self.self_attn.reset_length_norm(tgt_length / 2)
        if hasattr(self, "dec_global_attn"):
            if hasattr(self.dec_global_attn, "reset_length_norm"):
                self.dec_global_attn.reset_length_norm(tgt_length / 2)
        return

    def build_attn_gate(self, embed_dim: int):
        return AttnGate(embed_dim)

    def build_encoder_attention(
        self,
        embed_dim: int,
        args: Namespace,
        **kwargs,
    ):
        if ("dec" in args.length_norm) and (self.dec_type == "global"):
            average_length = 1.0
        elif kwargs.get("length_norm", False):
            average_length = 1.0
        else:
            average_length = None
        return WeightedAttention(
            embed_dim,
            args.decoder.attention_heads,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            average_length=average_length,
        )

    def build_self_attention(
        self,
        embed_dim: int,
        args: Namespace,
        *largs,
        **kwargs,
    ):
        if ("dec" in args.length_norm) and (self.dec_type == "global"):
            average_length = 1.0
        elif kwargs.get("length_norm", False):
            average_length = 1.0
        else:
            average_length = None
        return WeightedAttention(
            embed_dim,
            args.decoder.attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            average_length=average_length,
        )

    def update_mask(
        self,
        dec_local_mask: Tensor,
        dec_global_mask: Tensor,
        crs_local_mask: Tensor,
        crs_global_mask: Tensor,
        dec_ctx_mask: Tensor,
        crs_ctx_mask: Tensor,
        self_attn_mask: Tensor,
    ):
        # update self-attn mask
        if self.dec_type == "global":
            dec_local_mask = self_attn_mask
        elif (self.dec_type == "local") and (self_attn_mask is not None):
            dec_local_mask |= self_attn_mask.bool().unsqueeze(0)
        elif self.dec_type in ("hybrid", "stack", "selective", "legacy"):
            if self_attn_mask is not None:
                dec_local_mask |= self_attn_mask.bool().unsqueeze(0)
                dec_global_mask |= self_attn_mask.bool().unsqueeze(0)
            dec_global_mask.masked_fill_(dec_ctx_mask.unsqueeze(-1), False)
        elif self.dec_type == "group":
            if self_attn_mask is not None:
                dec_local_mask |= self_attn_mask.bool().unsqueeze(0)
            dec_global_mask = self_attn_mask
            dec_ctx_mask = None

        # update cross-attn mask
        if self.crs_type == "global":
            crs_local_mask = None
        elif self.crs_type == "group":
            crs_global_mask = None
            crs_ctx_mask = None
        elif self.crs_type in ("hybrid", "stack", "selective", "legacy"):
            crs_global_mask.masked_fill_(crs_ctx_mask.unsqueeze(-1), False)

        return (
            dec_local_mask,
            dec_global_mask,
            crs_local_mask,
            crs_global_mask,
            dec_ctx_mask,
            crs_ctx_mask,
        )

    def forward(
        self,
        x: Tensor,
        src_tags: Tensor,
        tgt_tags: Tensor,
        dec_local_mask: Tensor,
        dec_global_mask: Tensor,
        crs_local_mask: Tensor,
        crs_global_mask: Tensor,
        dec_ctx_mask: Tensor,
        crs_ctx_mask: Tensor,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        **kwargs,
    ):
        if need_head_weights:
            need_attn = True
        attn = {}

        src_length = kwargs.get("src_length", None)
        tgt_length = kwargs.get("tgt_length", None)

        (
            dec_local_mask,
            dec_global_mask,
            crs_local_mask,
            crs_global_mask,
            dec_ctx_mask,
            crs_ctx_mask,
        ) = self.update_mask(
            dec_local_mask,
            dec_global_mask,
            crs_local_mask,
            crs_global_mask,
            dec_ctx_mask,
            crs_ctx_mask,
            self_attn_mask,
        )

        # self attention
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if hasattr(self.self_attn, "length_norm"):
            addition_args = {"kv_length": kwargs.get("tgt_length", None)}
        else:
            addition_args = {}
        x_local, attn_weight = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=need_attn,
            attn_mask=dec_local_mask,
            **addition_args,
        )
        attn["dec"] = attn_weight

        # self global attention
        if self.dec_type in ("hybrid", "group", "legacy"):
            x_global, attn_weight = self.dec_global_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=need_attn,
                attn_mask=dec_global_mask,
                kv_length=tgt_length,
            )
            x = self.dec_attn_gate(x_local, x_global, dec_ctx_mask)
            attn[f"dec_{self.dec_type}"] = attn_weight
        elif self.dec_type == "stack":
            x_local = self.dropout_module(x_local)
            x_local = self.residual_connection(x_local, residual)
            if not self.normalize_before:
                x_local = self.self_attn_layer_norm(x_local)
            else:
                x_local = self.dec_global_norm(x_local)
            residual = x_local
            x_global, _ = self.dec_global_attn(
                query=x_local,
                key=x_local,
                value=x_local,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=need_attn,
                attn_mask=dec_global_mask,
                kv_length=tgt_length,
            )
            x = x_global.masked_fill(dec_ctx_mask.transpose(0, 1).unsqueeze(-1), 0.0)
            attn[f"dec_{self.dec_type}"] = attn_weight
        elif self.dec_type == "selective":
            x_global, attn_weight = self.dec_global_attn(
                query=x,
                key=x,
                value=x,
                key_tags=tgt_tags,
                query_tags=tgt_tags,
                key_padding_mask=self_attn_padding_mask,
                query_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                attn_mask=dec_global_mask,
            )
            x = self.dec_attn_gate(x_local, x_global, dec_ctx_mask)
        else:
            x = x_local

        # layer norm & residual connection
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            if hasattr(self, "dec_global_norm"):
                x = self.dec_global_norm(x)
            else:
                x = self.self_attn_layer_norm(x)

        # cross attention
        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        if hasattr(self.encoder_attn, "length_norm"):
            addition_args = {"kv_length": kwargs.get("src_length", None)}
        else:
            addition_args = {}
        x_local, attn_weight = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
            attn_mask=crs_local_mask,
            **addition_args,
        )
        attn[f"crs"] = attn_weight
        if self.crs_type in ("hybrid", "group", "legacy"):
            x_global, attn_weight = self.crs_global_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                attn_mask=crs_global_mask,
                kv_length=src_length,
            )
            x = self.crs_attn_gate(x_local, x_global, crs_ctx_mask)
            attn[f"crs_{self.crs_type}"] = attn_weight
        elif self.crs_type == "stack":
            x_local = self.dropout_module(x_local)
            x_local = self.residual_connection(x_local, residual)
            if not self.normalize_before:
                x_local = self.encoder_attn_layer_norm(x_local)
            else:
                x_local = self.crs_global_norm(x_local)
            residual = x_local
            x_global, attn_weight = self.crs_global_attn(
                query=x_local,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                attn_mask=crs_global_mask,
                kv_length=src_length,
            )
            x = x_global.masked_fill(crs_ctx_mask.transpose(0, 1).unsqueeze(-1), 0.0)
            attn[f"crs_{self.crs_type}"] = attn_weight
        elif self.crs_type == "selective":
            x_global, _ = self.crs_global_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_tags=src_tags,
                query_tags=tgt_tags,
                key_padding_mask=encoder_padding_mask,
                query_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                attn_mask=crs_global_mask,
            )
            x = self.crs_attn_gate(x_local, x_global, crs_ctx_mask)
        else:
            x = x_local

        # layer norm & residual connection
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            if hasattr(self, "crs_global_norm"):
                x = self.crs_global_norm(x)
            else:
                x = self.encoder_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        # feed forward function
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn, None
