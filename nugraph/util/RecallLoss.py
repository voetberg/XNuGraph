# as described in https://arxiv.org/abs/2106.14917

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.functional import recall


class RecallLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -1, num_classes=5):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        try:
            y = torch.argmax(target, dim=1)
        except IndexError:
            y = target

        # y_hat = torch.argmax(input, dim=1)
        weight = 1 - recall(
            input.to(dtype=torch.float),
            y,
            "multiclass",
            num_classes=self.num_classes,
            average=None,
            ignore_index=self.ignore_index,
        )
        ce = F.cross_entropy(input, y, reduction="none", ignore_index=self.ignore_index)
        loss = weight[y] * ce
        return loss.mean()
