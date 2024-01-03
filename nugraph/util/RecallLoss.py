# as described in https://arxiv.org/abs/2106.14917

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.functional import recall

class RecallLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -1, num_classes=5):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes=num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        y =  torch.argmax(target, dim=1) 
        y_hat = torch.argmax(input, dim=1)

        weight = 1 - recall(y_hat, y, "multiclass", num_classes=self.num_classes, average=None)
        # weight = 1 - recall(input, target, 'multiclass',
        #                     num_classes=input.size(1),
        #                     average='none',
        #                     ignore_index=self.ignore_index)

        ce = F.cross_entropy( target,input, reduction='none')
        loss = weight[y] * ce
        return loss.mean()