from functools import partialmethod

from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torchvision.models import vision_transformer

vision_transformer


def partial_cls(cls, *args, **kwargs):
    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass


class ResizedTransformer(Module):
    def __init__(self, network, size, *args, **kwargs):
        super().__init__()
        self.network = network(*args, **kwargs)
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=True)
        return self.network(x)


vit_l_32 = partial_cls(
    ResizedTransformer, vision_transformer.vit_l_32, (224, 224), num_classes=1000
)
vit_l_16 = partial_cls(
    ResizedTransformer, vision_transformer.vit_l_16, (224, 224), num_classes=1000
)
vit_h_14 = partial_cls(
    ResizedTransformer, vision_transformer.vit_h_14, (224, 224), num_classes=1000
)
vit_b_16 = partial_cls(
    ResizedTransformer, vision_transformer.vit_b_16, (224, 224), num_classes=1000
)
vit_b_32 = partial_cls(
    ResizedTransformer, vision_transformer.vit_b_32, (224, 224), num_classes=1000
)
