from fairseq.tasks.translation import TranslationConfig
from typing import Optional, Tuple
from dataclasses import field, dataclass
from fairseq.dataclass import ChoiceEnum


AVAILABLE_DATA_TYPE = [
    "seg2seg",
    "hybrid",
    "sent2sent",
    "doc2sent",
    "context",
    "divide",
    "golden",
]


@dataclass
class DocumentConfig(TranslationConfig):
    data_type: ChoiceEnum(AVAILABLE_DATA_TYPE) = field(
        default="seg2seg",
        metadata={
            "help": "the type of the data in __get_item__.\n"
            "seg2seg: return a segment\n"
            "hybrid: build segments using multi-resolution strategy\n"
            "sent2sent: act as language pair dataset\n"
            "doc2sent: build segmente-level source and sentence-level target\n"
            "context: return a sentence with its context\n"
            "divide: build splited sentences using divide & rule strategy"
        },
    )
    # filter long sentences
    max_sent_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "pre-filter sentence by sentence length",
        },
    )
    # used in seg2seg/sampled data_type
    segment_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "length of the segment, 0 for infinite",
        },
    )
    sentence_num: Optional[int] = field(
        default=0,
        metadata={
            "help": "max sentence num of the segments, 0 for infinite",
        },
    )
    sampled_segment: Optional[bool] = field(
        default=False,
        metadata={
            "help": "sampling segments from the training dataset for each epoch",
        },
    )
    adaptive_sample: Optional[bool] = field(
        default=False,
        metadata={
            "help": "sample segments from the dataset for each epoch",
        },
    )
    start_sample_epoch: Optional[int] = field(
        default=20,
        metadata={
            "help": "start sampling from this epoch",
        },
    )
    allow_mixup: Optional[bool] = field(
        default=False,
        metadata={
            "help": "allow mixup the documents in the training dataset",
        },
    )
    # used in mr data_type
    hybrid_sents: Optional[Tuple[int]] = field(
        default=(1, 2, 4, 8, 999),
        metadata={
            "help": "max number for multi-resolution training",
        },
    )
    # used in doc2sent and context data_type
    context_num: Optional[Tuple[int]] = field(
        default=(3, 3),
        metadata={
            "help": "number of context sentences (before, after)",
        },
    )
    concate_context: Optional[bool] = field(
        default=False,
        metadata={
            "help": "concate context sentences",
        },
    )
    # used in seg2seg, doc2sent, mr data_type
    use_tags: Optional[bool] = field(
        default=False,
        metadata={
            "help": "provide tags in net_input",
        },
    )
    # used in seg2seg, mr data_type
    use_mask: Optional[bool] = field(
        default=False,
        metadata={
            "help": "provide local/global mask in net_input",
        },
    )
    # used in all data_type
    word_drop: float = field(
        default=0.0,
        metadata={
            "help": "apply word_drop to the data",
        },
    )
    word_drop_epoch: int = field(
        default=40,
        metadata={
            "help": "apply word_drop for first N epochs",
        },
    )
    mask_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "token used for word_drop",
        },
    )
    # used in segment generator
    force_decode: Optional[bool] = field(
        default=False,
        metadata={
            "help": "force generate to ensure the sentence number match",
        },
    )
    disable_incremental: Optional[bool] = field(
        default=False,
        metadata={
            "help": "disable incremental decoding",
        },
    )
    partial_load: Optional[bool] = field(
        default=False,
        metadata={
            "help": "allow missing parameter when loading",
        },
    )
    allow_longer: Optional[bool] = field(
        default=False,
        metadata={
            "help": "do not filter indices by size",
        }
    )
    golden_context: Optional[bool] = field(
        default=False,
        metadata={
            "help": "use golden context for evaluation",
        }
    )
    slide_decode: Optional[bool] = field(
        default=False,
        metadata={
            "help": "slide decoding for long sentences",
        }
    )
    context_window: Optional[int] = field(
        default=0,
        metadata={
            "help": "ensures that every evaluated sentence has access to a context of at least this size, if possible"
        }
    )
    forward_encoder_once: Optional[bool] = field( # TODO: support this
        default=False,
        metadata={
            "help": "forward encoder only once",
        }
    )
    # used for load mBART
    finetune_mbart: Optional[bool] = field(
        default=False,
        metadata={
            "help": "finetune from pretrained mBART",
        },
    )
    langs: Optional[str] = field(
        default=None,
        metadata={
            "help": "supported langs (used only in mBART)",
        },
    )
    profile_model: Optional[bool] = field(
        default=False,
        metadata={
            "help": "make model profile using deepspeed",
        },
    )
