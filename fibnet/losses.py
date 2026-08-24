from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        targets = targets.flatten(1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score.mean()


class TverskyLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.4, beta: float = 0.6, smooth: float = 1.0
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)

        true_positive = (probabilities * targets).sum(dim=1)
        false_positive = (probabilities * (1.0 - targets)).sum(dim=1)
        false_negative = ((1.0 - probabilities) * targets).sum(dim=1)
        score = (true_positive + self.smooth) / (
            true_positive
            + (self.alpha * false_positive)
            + (self.beta * false_negative)
            + self.smooth
        )
        return 1.0 - score.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.4,
        tversky_beta: float = 0.6,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
            )
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        tversky_loss = self.tversky(logits, targets)
        return (
            (self.bce_weight * bce_loss)
            + (self.dice_weight * dice_loss)
            + (self.tversky_weight * tversky_loss)
        )
