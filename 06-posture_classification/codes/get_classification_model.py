from torchvision import models

from lib.networks import SaccpaRes, eca

classification_models = {
    "ResNet34": models.resnet34,
    "Resnet50": models.resnet50,
    "Resnet101": models.resnet101,
    "Resnet152": models.resnet152,
    "EfficientNetB0": models.efficientnet_b0,
    "EfficientNetB2": models.efficientnet_b2,
    "EfficientNetB4": models.efficientnet_b4,
    "EfficientNetB7": models.efficientnet_b7,
    "ECA34": eca.eca_resnet34,
    "ECA50": eca.eca_resnet50,
    "ECA101": eca.eca_resnet101,
    "ECA152": eca.eca_resnet152,
    "SACCPA34": SaccpaRes.saccpa_resnet34,
    "SACCPA50": SaccpaRes.saccpa_resnet50,
    "SACCPA101": SaccpaRes.saccpa_resnet101,
    "SACCPA152": SaccpaRes.saccpa_resnet152,
}
