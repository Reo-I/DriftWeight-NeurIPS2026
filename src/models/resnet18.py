"""CIFAR-style ResNet-18 model for CIFAR-10."""

import torch.nn as nn
import torchvision


def ResNet18(num_classes: int = 10) -> nn.Module:
    """Build ResNet-18 adapted for CIFAR-10 resolution.

    Uses the standard torchvision ResNet-18 backbone with CIFAR-specific stem:
    3x3 conv (stride 1) and no max-pooling.
    """
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
