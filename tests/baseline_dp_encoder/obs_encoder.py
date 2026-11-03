import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18ObsEncoder(nn.Module):
    """
    Encode one RGB observation into a feature vector.

    Input:
        (B, 3, H, W)

    Output:
        (B, feature_dim)
    """

    def __init__(
        self,
        pretrained: bool = False,
    ):
        super().__init__()

        if pretrained:
            from torchvision.models import (
                ResNet18_Weights,
            )

            backbone = resnet18(
                weights=ResNet18_Weights.DEFAULT
            )

        else:
            backbone = resnet18(
                weights=None
            )

        # ResNet18 classifier input = 512
        self.feature_dim = (
            backbone.fc.in_features
        )

        # Remove classification layer
        backbone.fc = nn.Identity()

        self.backbone = backbone

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "Expected image tensor with shape "
                f"(B, C, H, W), got {x.shape}"
            )

        return self.backbone(x)

    def output_shape(self):
        return (self.feature_dim,)