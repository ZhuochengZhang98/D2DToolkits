from fairseq.optim import register_optimizer
from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig

@register_optimizer("null_optim", FairseqAdamConfig)
class NullOptim(FairseqAdam):
    def step(self, closure=None, **kwargs):
        if closure is not None:
            return closure()
        else:
            return None