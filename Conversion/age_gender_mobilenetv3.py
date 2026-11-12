from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


class AgeGenderMobileNetV3(nn.Module):
    def __init__(self, variant: str = "large", pretrained: bool = True, age_bins: int = 101, dropout: float = 0.2) -> None:
        super().__init__()
        if variant == "large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base = mobilenet_v3_large(weights=weights)
        elif variant == "small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = mobilenet_v3_small(weights=weights)
        else:
            raise ValueError(f"unsupported MobileNetV3 variant: {variant}")

        in_features = base.classifier[0].in_features
        self.features = base.features
        self.avgpool = base.avgpool
        self.neck = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
        )
        self.gender_head = nn.Linear(512, 2)
        self.age_head = nn.Linear(512, age_bins)
        self.register_buffer("age_values", torch.arange(age_bins, dtype=torch.float32), persistent=False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(image)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.neck(x)
        gender_logits = self.gender_head(x)
        age_logits = self.age_head(x)
        age_probs = torch.softmax(age_logits, dim=1)
        age = (age_probs * self.age_values.to(age_probs.device)).sum(dim=1)
        return {
            "gender_logits": gender_logits,
            "age_logits": age_logits,
            "age": age,
        }
