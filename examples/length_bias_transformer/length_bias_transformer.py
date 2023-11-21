import logging
from argparse import ArgumentParser, Namespace
from typing import Optional

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, base_architecture
from omegaconf import DictConfig

from .length_bias_decoder import LengthBiasDecoder
from .length_bias_encoder import LengthBiasEncoder


AVAILABLE_LAYER_TYPE = [
    "local",
    "global",
    "hybrid",
    "stack",
    "group",
]


logger = logging.getLogger(__name__)


@register_model("length_bias_transformer")
class LengthBiasTransformer(TransformerModel):
    @classmethod
    def add_args(cls, parser: ArgumentParser):
        super().add_args(parser)
        parser.add_argument(
            "--enc-type",
            nargs="+",
            choices=AVAILABLE_LAYER_TYPE,
            type=str,
            help="layer type of encoder self-attention",
        )
        parser.add_argument(
            "--dec-type",
            nargs="+",
            choices=AVAILABLE_LAYER_TYPE,
            type=str,
            help="layer type of decoder self-attention",
        )
        parser.add_argument(
            "--crs-type",
            nargs="+",
            choices=AVAILABLE_LAYER_TYPE,
            type=str,
            help="layer type of decoder cross-attention",
        )
        parser.add_argument(
            "--seg-emb",
            default=False,
            action="store_true",
            help="enable segment embedding",
        )
        parser.add_argument(
            "--freeze",
            default=False,
            action="store_true",
            help="freeze the parameters of the model",
        )
        parser.add_argument(
            "--length-norm",
            type=str,
            nargs="*",
            default=[],
            choices=["enc", "dec", "crs"],
            help="use length normalization in attention",
        )
        parser.add_argument(
            "--adjust-length-norm",
            default=False,
            action="store_true",
            help="adjust length norm to match the segment length",
        )
        return

    def __init__(self, args, encoder: LengthBiasEncoder, decoder: LengthBiasDecoder):
        super().__init__(args, encoder, decoder)
        self.partial_load = getattr(args, "partial_load", False)
        self.spliter_src = encoder.dictionary.eos_index
        self.spliter_tgt = decoder.dictionary.eos_index
        self.adjust_length_norm = args.adjust_length_norm
        if args.freeze:
            self.encoder.embed_tokens.requires_grad_(False)
            self.decoder.embed_tokens.requires_grad_(False)
            self.decoder.output_projection.requires_grad_(False)
        return

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return LengthBiasEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return LengthBiasDecoder(args, tgt_dict, embed_tokens)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs,
    ):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            src_tags=kwargs.get("src_tags", None),
            local_mask=kwargs.get("src_local_mask", None),
            global_mask=kwargs.get("src_global_mask", None),
            sent_embs=kwargs.get("sent_embs", None),
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            src_tags=kwargs.get("src_tags", None),
            tgt_tags=kwargs.get("tgt_tags", None),
            dec_local_mask=kwargs.get("dec_local_mask", None),
            dec_global_mask=kwargs.get("dec_global_mask", None),
            crs_local_mask=kwargs.get("crs_local_mask", None),
            crs_global_mask=kwargs.get("crs_global_mask", None),
        )
        return decoder_out

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        strict = False if self.partial_load else True
        missing_keys, unexpected_keys = super().load_state_dict(
            state_dict, strict, model_cfg, args
        )
        if self.partial_load:
            for key in missing_keys:
                logger.warning(f"Missing key: {key}")
            for key in unexpected_keys:
                logger.warning(f"Unexpected key: {key}")
        return missing_keys, unexpected_keys

    def set_seg_length(self, src_seg_length: float, tgt_seg_length: float):
        if self.adjust_length_norm:
            logger.info(f"set encoder segment length to {src_seg_length}")
            logger.info(f"set decoder segment length to {tgt_seg_length}")
            self.encoder.set_seg_length(src_seg_length)
            self.decoder.set_seg_length(src_seg_length, tgt_seg_length)
        else:
            src_len = self.encoder.segment_length
            if src_len is not None:
                if src_len == 1.0:
                    self.encoder.set_seg_length(src_seg_length)
                    logger.info(f"set encoder segment length to {src_seg_length}")
                else:
                    logger.info(f"encoder segment length {src_len}")
            src_len, tgt_len = self.decoder.segment_length
            if src_len is not None:
                if src_len == 1.0:
                    self.decoder.set_src_length(src_seg_length)
                    logger.info(f"set decoder source length to {src_seg_length}")
                else:
                    logger.info(f"decoder source length {src_len}")
            if tgt_len is not None:
                if tgt_len == 1.0:
                    self.decoder.set_tgt_length(tgt_seg_length)
                    logger.info(f"set decoder target length to {tgt_seg_length}")
                else:
                    logger.info(f"decoder target length {src_len}")
        return


def base(args: Namespace):
    base_architecture(args)
    # set default parameters
    args.enc_type = getattr(args, "enc_type", ["global"])
    args.dec_type = getattr(args, "dec_type", ["global"])
    args.crs_type = getattr(args, "crs_type", ["global"])
    if len(args.enc_type) == 1:
        args.enc_type = args.enc_type * args.encoder_layers
    if len(args.dec_type) == 1:
        args.dec_type = args.dec_type * args.decoder_layers
    if len(args.crs_type) == 1:
        args.crs_type = args.crs_type * args.decoder_layers
    assert len(args.enc_type) == args.encoder_layers
    assert len(args.dec_type) == args.decoder_layers
    assert len(args.crs_type) == args.decoder_layers
    args.need_tags = getattr(args, "need_tags", True)
    args.length_norm = getattr(args, "length_norm", [])
    if args.length_norm == True:
        args.length_norm = ["enc", "dec", "crs"]
    if args.length_norm == False:
        args.length_norm = []
    args.adjust_length_norm = getattr(args, "adjust_length_norm", False)
    return


@register_model_architecture("length_bias_transformer", "global_transformer")
def sat_base(args: Namespace):
    args.enc_type = getattr(args, "enc_type", ["global"])
    args.dec_type = getattr(args, "dec_type", ["global"])
    args.crs_type = getattr(args, "crs_type", ["global"])
    args.need_tags = True if args.seg_emb else False
    base(args)


@register_model_architecture("length_bias_transformer", "sent_transformer")
def sat_base(args: Namespace):
    args.enc_type = getattr(args, "enc_type", ["local"])
    args.dec_type = getattr(args, "dec_type", ["local"])
    args.crs_type = getattr(args, "crs_type", ["local"])
    base(args)


@register_model_architecture("length_bias_transformer", "stack_transformer")
def sat_base(args: Namespace):
    args.enc_type = getattr(args, "enc_type", ["stack"])
    args.dec_type = getattr(args, "dec_type", ["stack"])
    args.crs_type = getattr(args, "crs_type", ["stack"])
    base(args)


@register_model_architecture("length_bias_transformer", "hybrid_transformer")
def sat_base(args: Namespace):
    args.enc_type = getattr(args, "enc_type", ["hybrid"])
    args.dec_type = getattr(args, "dec_type", ["hybrid"])
    args.crs_type = getattr(args, "crs_type", ["hybrid"])
    base(args)


@register_model_architecture("length_bias_transformer", "g_transformer")
def sat_base(args: Namespace):
    args.enc_type = getattr(args, "enc_type", ["local"] * 4 + ["group"] * 2)
    args.dec_type = getattr(args, "dec_type", ["local"] * 4 + ["group"] * 2)
    args.crs_type = getattr(args, "crs_type", ["local"] * 4 + ["group"] * 2)
    base(args)
