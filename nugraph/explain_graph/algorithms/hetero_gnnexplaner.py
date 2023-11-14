from typing import Any, Dict, Optional, Union
import math 
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import Explainer, Explanation, ExplainerAlgorithm, GNNExplainer, HeteroExplanation 
from torch_geometric.explain.algorithm.utils import set_masks
import pandas as pd 
from torch_geometric.data import HeteroData, DataLoader, batch
from torch_geometric.typing import EdgeType, NodeType
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

    def __call__(self, graph) -> Explanation | HeteroExplanation:
        x, edge_index, target, index = None, None, None, None
        kwargs={"graph":graph}
        explaination =  super().__call__(x, edge_index, target=target, index=index, **kwargs)
        graph = next(iter(graph))
        if 'edge_mask' in explaination: 
            graph['edge_mask'] = explaination['edge_mask']
        if "node_mask" in explaination: 
            graph['node_mask'] = explaination['node_mask']
        return graph

class HeteroGNNExplainer(GNNExplainer): 
    def __init__(self, epochs: int = 100, lr: float = 0.01, plane='u', single_plane=True, **kwargs):
        super().__init__(epochs, lr, **kwargs)
        self.plane = plane
        self.single_plane = single_plane
        if not self.single_plane: 
            assert type(plane) == list

    def forward(self, model, x, edge_index=None, node_index=None, **kwargs):
        graph = next(iter(kwargs['graph']))

        model.nexus_net.explain = False 

        for decoder in model.decoders: 
            decoder.explain = False 

        self._train(model, graph, node_index)

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
    
    def assign_planar_masks(self, graph, plane): 
        x_mask = graph[plane]['x']
        edge_index_mask = graph[plane, 'plane', plane]['edge_index'].to(torch.float)

        (N, F), E = x_mask.size(), edge_index_mask.size(1)
        node_mask = Parameter(torch.randn(N, F) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * math.sqrt(2.0 / (2 * N))
        edge_mask = Parameter(torch.randn(E) * std)

        parameters = []
        if self.node_mask is not None:
            parameters.append(node_mask)

        if self.edge_mask is not None:
            parameters.append(edge_mask)
        
        return parameters, node_mask, edge_mask

    def update_planar_graph(self, graph, plane): 
        x = graph[plane]['x']
        h = x if self.node_mask is None else x * self.node_mask.sigmoid()
        graph[plane]['x'] = h 

        # Update the edge masks 
        edges =  graph[plane, 'plane', plane]['edge_index'].to(torch.float)
        edge_mask_h = edges if self.edge_mask is None else edges * self.edge_mask.sigmoid()
        graph[plane, 'plane', plane]['edge_index'] = edge_mask_h

        return graph

    def assign_nexus_masks(self, graph): 
        print(graph)
        self.node_mask = {}
        self.edge_mask = {}
        all_parameters = []

        ## planar masks 
        for plane in self.plane: 
            ## planar masks 
            parameters, node_mask, edge_mask = self.assign_planar_masks(graph, plane)
            all_parameters+=parameters
            self.node_mask[plane] = node_mask
            self.edge_mask[plane] = edge_mask
            
            ## nexus masks 
            nexus_edge = graph[(plane, 'nexus', 'sp')]['edge_index'].to(torch.float)
            N, _ = node_mask.size()
            E = nexus_edge.size(1)
            std = torch.nn.init.calculate_gain('relu') * math.sqrt(2.0 / (2 * N))
            edge_mask = Parameter(torch.randn(E) * std)

            self.edge_mask[f"{plane}_nexus"] = edge_mask
            all_parameters.append(edge_mask)

        return all_parameters
    
    def update_nexus_masks(self, graph): 
        for plane in self.plane: 
            plane_node_mask = ""
            plane_edge_planar_mask = ""
            plane_edge_nexus_mask = ""
            

    def _train(self, model, graph, node_index=None, **kwargs):
        graph.requires_grad=True

        if self.single_plane: 
            parameters, self.node_mask, self.edge_mask = self.assign_planar_masks(graph, self.plane)
        else: 
            parameters = self.assign_nexus_masks(graph)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)
 
        for i in range(self.epochs):
            optimizer.zero_grad()

            if self.single_plane: 
                graph = self.update_planar_graph(graph, self.plane)
            else: 
                graph = self.update_nexus_masks(graph)

            model.step(graph)

            y = graph[self.plane]['y_semantic'].to(torch.float)
            y_hat = torch.argmax(graph[self.plane]['x_semantic'], dim=-1).to(torch.float)

            assert len(y) == len(y_hat) ## Personal check that things are not weirdly transposed

            if node_index is not None: 
                y = y[node_index]
                y_hat = y_hat[node_index]
            
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

