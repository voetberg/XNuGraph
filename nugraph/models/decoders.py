from typing import Any, Callable

from abc import ABC

from torch import Tensor, tensor, cat
import torch.nn as nn
from torch_geometric.nn.aggr import SoftmaxAggregation, LSTMAggregation

import torchmetrics as tm

import matplotlib.pyplot as plt
import seaborn as sn
import math

from .linear import ClassLinear
from ..util import RecallLoss, LogCoshLoss

class DecoderBase(nn.Module, ABC):
    '''Base class for all NuGraph decoders'''
    def __init__(self,
                 name: str,
                 planes: list[str],
                 classes: list[str],
                 loss_func: Callable,
                 weight: float,
                 temperature: float = 0.):
        super().__init__()
        self.name = name
        self.planes = planes
        self.classes = classes
        self.loss_func = loss_func
        self.weight = weight
        self.temp = nn.Parameter(tensor(temperature))
        self.confusion = nn.ModuleDict()

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        raise NotImplementedError

    def loss(self,
             batch,
             stage: str,
             confusion: bool = False):
        x, y = self.arrange(batch)
        metrics = self.metrics(x, y, stage)
        w = self.weight * (-1 * self.temp).exp()
        loss = w * self.loss_func(x, y) + self.temp
        metrics[f'loss_{self.name}/{stage}'] = loss
        if stage == 'train':
            metrics[f'temperature/{self.name}'] = self.temp
        for cm in self.confusion.values():
            cm.update(x, y)
        return loss, metrics

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        '''Produce confusion matrix at end of epoch'''
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(confusion,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel('Assigned label')
        plt.ylabel('True label')
        return fig

    def on_epoch_end(self,
                     logger: 'pl.loggers.TensorBoardLogger',
                     stage: str,
                     epoch: int) -> None:
        if not logger: return
        for name, cm in self.confusion.items():
            logger.experiment.add_figure(
                f'{name}/{stage}',
                self.draw_confusion_matrix(cm),
                global_step=epoch)
            cm.reset()

class SemanticDecoder(DecoderBase):
    """NuGraph semantic decoder module.

    Convolve down to a single node score per semantic class for each 2D graph,
    node, and remove intermediate node stores from data object.
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__('semantic',
                         planes,
                         semantic_classes,
                         RecallLoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            'task': 'multiclass',
            'num_classes': len(semantic_classes),
            'ignore_index': -1
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_semantic_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_semantic_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = ClassLinear(node_features, 1, len(semantic_classes))

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return { 'x_semantic': { p: self.net[p](x[p]).squeeze(dim=-1) for p in self.planes } }

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        x = cat([batch[p].x_semantic for p in self.planes], dim=0)
        y = cat([batch[p].y_semantic for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'recall_semantic/{stage}': self.recall(x, y),
            f'precision_semantic/{stage}': self.precision(x, y)
        }

class FilterDecoder(DecoderBase):
    """NuGraph filter decoder module.

    Convolve down to a single node score, to identify and filter out
    graph nodes that are not part of the primary physics interaction
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__('filter',
                         planes,
                         ('noise', 'signal'),
                         nn.BCELoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            'task': 'binary'
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_filter_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_filter_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        num_features = len(semantic_classes) * node_features
        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid())

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]):
        return { 'x_filter': { p: self.net[p](x[p].flatten(start_dim=1)).squeeze(dim=-1) for p in self.planes }}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        x = cat([batch[p].x_filter for p in self.planes], dim=0)
        y = cat([(batch[p].y_semantic!=-1).float() for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'recall_filter/{stage}': self.recall(x, y),
            f'precision_filter/{stage}': self.precision(x, y)
        }

class EventDecoder(DecoderBase):
    '''NuGraph event decoder module.

    Convolve graph node features down to a single classification score
    for the entire event
    '''
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str],
                 event_classes: list[str]):
        super().__init__('event',
                         planes,
                         event_classes,
                         RecallLoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            'task': 'multiclass',
            'num_classes': len(event_classes)
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_event_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_event_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.pool = nn.ModuleDict()
        for p in planes:
            self.pool[p] = SoftmaxAggregation(learn=True)
        self.net = nn.Sequential(
            nn.Linear(in_features=len(planes) * len(semantic_classes) * node_features,
                      out_features=len(event_classes)))

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]):
        x = [ pool(x[p].flatten(1), batch[p]) for p, pool in self.pool.items() ]
        return { 'x': { 'evt': self.net(cat(x, dim=1)) }}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        return batch['evt'].x, batch['evt'].y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'recall_event/{stage}': self.recall(x, y),
            f'precision_event/{stage}': self.precision(x, y)
        }

class VertexDecoder(DecoderBase):
    """
    """
    def __init__(self,
                 node_features: int,
                 vertex_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__('vertex',
                         planes,
                         semantic_classes,
                         LogCoshLoss(),
                         weight=1.,
                         temperature=5.)
        in_features = len(semantic_classes) * node_features
        self.lstm = nn.ModuleDict()
        for p in planes:
            self.lstm[p] = LSTMAggregation(in_channels=in_features,
                                           out_channels=vertex_features)
        self.net = nn.Sequential(
            nn.Linear(in_features=len(planes) * vertex_features,
                      out_features=vertex_features),
            nn.ReLU(),
            nn.Linear(in_features=vertex_features,
                      out_features=3))

    def forward(self, x: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str,dict[str, Tensor]]:
        x = [ net(x[p].flatten(1), index=batch[p]) for p, net in self.lstm.items() ]
        x = cat(x, dim=1)
        return { 'x_vtx': { 'evt': self.net(x) }}

    def arrange(self, batch):
        x = batch['evt'].x_vtx
        y = batch['evt'].y_vtx
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str):
        xyz = (x-y).abs().mean(dim=0)
        return {
            f'vertex-resolution-x/{stage}': xyz[0],
            f'vertex-resolution-y/{stage}': xyz[1],
            f'vertex-resolution-z/{stage}': xyz[2],
            f'vertex-resolution/{stage}': xyz.square().sum().sqrt()
        }
