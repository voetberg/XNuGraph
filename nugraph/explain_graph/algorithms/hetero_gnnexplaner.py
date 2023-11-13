from typing import Any, Dict, Optional, Union
import math 
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import Explainer, Explanation, ExplainerAlgorithm, GNNExplainer 
from torch_geometric.explain.algorithm.utils import set_masks
import pandas as pd 
from torch_geometric.data import HeteroData, DataLoader
from nugraph.explain_graph.load import Load
from nugraph import util
import pytorch_lightning as pl



class HeteroExplainer(Explainer): 
    def __init__(self, model: torch.nn.Module, algorithm: ExplainerAlgorithm, explanation_type, edge_mask_type, model_config, node_mask_type = None, threshold_config = None):
        super().__init__(model, algorithm, explanation_type, model_config, node_mask_type, edge_mask_type, threshold_config)

    def get_prediction(self, *args, **kwargs) -> Tensor:
        x, plane_edge, nexus_edge, nexus, batch = Load.unpack(kwargs['graph'])
        return self.model(x, plane_edge, nexus_edge, nexus, batch)

    def get_target(self, prediction: HeteroData) -> Tensor:
        preds = prediction['x_semantic']
        target = {}
        for plane in preds.keys(): 
            target[plane] = pd.Categorical(prediction["x_semantic"][plane][0].detach()).codes
        return target


class HeteroGNNExplainer(GNNExplainer): 
    def __init__(self, epochs: int = 100, lr: float = 0.01, plane='u', **kwargs):
        super().__init__(epochs, lr, **kwargs)
        self.plane = plane

    def forward(self, model, x, edge_index, **kwargs):
        graph = next(iter(kwargs['graph']))
        model.nexus_net.explain = False 
        for decoder in model.decoders: 
            decoder.explain = False 
        self._train(model, graph)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )

        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)
        explainer = Explanation(node_mask=node_mask, edge_mask=edge_mask)
        
        explainer[self.plane] = {}
        explainer[self.plane]['pos'] = graph[self.plane]['pos']
        explainer[self.plane]['pred_label'] = graph[self.plane]['x_semantic']
        explainer[self.plane]['sem_label'] = graph[self.plane]['y_semantic']
        explainer[self.plane]['x']=graph[self.plane]['x']

        return explainer
    
    def _train(self, model, graph,  **kwargs):

        model.step(graph)

       
        ## Use only a single plane - the x tensor used for analysis is different than the tensor used for the forward prediction
        x_mask = graph[self.plane]['x']
        x_mask.requires_grad = True
        edge_index_mask = graph[self.plane, 'plane', self.plane]['edge_index'].to(torch.float)
        edge_index_mask.requires_grad = True

        (N, F), E = x_mask.size(), edge_index_mask.size(1)
        self.node_mask = Parameter(torch.randn(N, 1) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * math.sqrt(2.0 / (2 * N))
        self.edge_mask = Parameter(torch.randn(E) * std)


        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index_mask, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            y = graph[self.plane]['y_semantic'].to(torch.float)
            y.requires_grad = True
            y_hat = torch.argmax(graph[self.plane]['x_semantic'], dim=-1).to(torch.float)
            y_hat.requires_grad = True

            assert len(y) == len(y_hat) ## Personal check that things are not weirdly transposed
            
            loss = self._loss(y_hat, y)
            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).

            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                    "features. Please make sure that node "
                                    "features are used inside the model or "
                                    "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                    "Please make sure that edges are used "
                                    "via message passing inside the model or "
                                    "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _loss(self, y_hat, y): 

        loss = self._loss_multiclass_classification(y_hat, y)

        m = self.edge_mask[self.hard_edge_mask].sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + self.coeffs['EPS']) - (
            1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_mask[self.hard_node_mask].sigmoid()
        node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
        ent = -m * torch.log(m + self.coeffs['EPS']) - (
            1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss