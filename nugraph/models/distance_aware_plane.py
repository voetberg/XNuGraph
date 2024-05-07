from typing import Any, Callable

from torch import Tensor, cat
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import MessagePassing

from .linear import ClassLinear


class EdgeMessagePassing2D(MessagePassing):
    propagate_type = {"x": Tensor}

    def __init__(
        self,
        in_features: int,
        planar_features: int,
        edge_features: int,
        num_classes: int,
        aggr: str = "add",
    ):
        super().__init__(node_dim=0, aggr=aggr)

        self.edge_net = nn.Sequential(
            ClassLinear(2 * (in_features + planar_features), 1, num_classes),
            nn.Softmax(dim=1),
        )

        self.edge_features_net = nn.Sequential(
            ClassLinear(edge_features, 1, num_classes), nn.Softmax(dim=1)
        )

        self.node_net = nn.Sequential(
            ClassLinear(
                2 * (in_features + planar_features), planar_features, num_classes
            ),
            nn.Tanh(),
            ClassLinear(planar_features, planar_features, num_classes),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        return self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.shape[0], x.shape[0])
        )

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor):
        node_features = cat((x_i, x_j), dim=-1)
        try:
            edge_encoding = self.edge_features_net(edge_attr).detach()
        except RuntimeError:
            print("WARNING: edge attr could not be encoded")
            edge_encoding = 1
        return self.edge_net(node_features).detach() * edge_encoding * x_j

    def update(self, aggr_out: Tensor, x: Tensor):
        return self.node_net(cat((x, aggr_out), dim=-1))


class DistanceAwarePlaneNet(nn.Module):
    """Module to convolve within each detector plane"""

    def __init__(
        self,
        in_features: int,
        planar_features: int,
        edge_features: int,
        num_classes: int,
        planes: list[str],
        aggr: str = "add",
        checkpoint: bool = True,
    ):
        super().__init__()

        self.checkpoint = checkpoint

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = EdgeMessagePassing2D(
                in_features, planar_features, edge_features, num_classes, aggr
            )

    def ckpt(self, fn: Callable, *args) -> Any:
        if self.checkpoint and self.training:
            return checkpoint(fn, *args)
        else:
            return fn(*args)

    def forward(
        self,
        x: dict[str, Tensor],
        edge_index: dict[str, Tensor],
        edge_attr: dict[str, Tensor],
    ) -> None:
        for p in self.net:
            print(edge_attr[p].shape)
            x[p] = self.ckpt(self.net[p], x[p], edge_index[p], edge_attr[p])
