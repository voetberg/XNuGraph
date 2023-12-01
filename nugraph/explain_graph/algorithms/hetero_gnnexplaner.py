from typing import Any, Dict, Optional, Union
from torch_geometric.explain.config import ThresholdConfig, ThresholdType
import copy

import math 
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import Explainer, Explanation, ExplainerAlgorithm, GNNExplainer, HeteroExplanation 
import pandas as pd 
from torch_geometric.data import HeteroData
from torch_geometric.explain.config import MaskType
from torch_geometric.typing import EdgeType, NodeType
from nugraph.explain_graph.load import Load
from nugraph.explain_graph.masking_utils import get_masked_graph


class HeteroExplainer(Explainer): 
    def __init__(self, model: torch.nn.Module, algorithm: ExplainerAlgorithm, explanation_type, model_config,edge_mask_type=None, node_mask_type = None, threshold_config = None):
        super().__init__(model, algorithm, explanation_type, model_config, node_mask_type, edge_mask_type, threshold_config)

    def get_prediction(self, *args, **kwargs) -> Tensor:
        x, plane_edge, nexus_edge, nexus, batch = Load.unpack(kwargs['graph'])
        return self.model(x, plane_edge, nexus_edge, nexus, batch)

    def get_target(self, prediction: HeteroData) -> Tensor:
        preds = prediction['x_semantic']
        target = {}
        for plane in preds.keys(): 
            target[plane] =torch.argmax(prediction["x_semantic"][plane].detach(), dim=-1)
        return target

    def get_masked_prediction(self, x: Tensor | Dict[NodeType, Tensor], edge_index: Tensor | Dict[EdgeType, Tensor], node_mask: Tensor | Dict[NodeType, Tensor] | None = None, edge_mask: Tensor | Dict[EdgeType, Tensor] | None = None, **kwargs) -> Tensor:
        masked_graph = kwargs['graph']
        masked_graph = get_masked_graph(masked_graph, node_mask, edge_mask)
        out = self.get_prediction(graph=masked_graph)
        return out

    def __call__(self, graph) -> Explanation | HeteroExplanation:
        x, edge_index, target, index = None, None, None, None
        kwargs={"graph":graph}

        try: 
            explaination =  super().__call__(x, edge_index, target=target, index=index, **kwargs)
        except AttributeError: 
            explaination = self.hetero_call(x, edge_index, target=target, index=index, **kwargs)

        ## Checking for the types to return for explaination

        if 'edge_mask' in explaination: 
            graph['edge_mask'] = explaination['edge_mask']
        if "node_mask" in explaination: 
            graph['node_mask'] = explaination['node_mask']
        return explaination

    def hetero_call(self, x, edge_index, target, index, **kwargs): 
        prediction: Optional[Tensor] = None

        prediction = self.get_prediction(x, edge_index, **kwargs)
        target = self.get_target(prediction)

        if isinstance(index, int):
            index = torch.tensor([index])

        training = self.model.training
        self.model.eval()

        explanation = self.algorithm(
            self.model,
            x,
            edge_index,
            target=target,
            index=index,
            **kwargs,
        )

        self.model.train(training)

        # Add explainer objectives to the `Explanation` object:
        explanation._model_config = self.model_config
        explanation._model_args = kwargs
        explanation.prediction = prediction
        explanation.target = target
        explanation.index = index

        for key, arg in kwargs.items():  # Add remaining `kwargs`:
            if isinstance(arg, dict):
                # Keyword arguments are likely named `{attr_name}_dict`
                # while we only want to assign the `{attr_name}` to the
                # `HeteroExplanation` object:
                key = key[:-5] if key.endswith('_dict') else key
                explanation.set_value_dict(key, arg)
            else:
                explanation[key] = arg
        
        return explanation.threshold(self.threshold_config)


class HeteroGNNExplainer(GNNExplainer): 
    def __init__(self, epochs: int = 100, lr: float = 0.01, plane='u', **kwargs):
        super().__init__(epochs, lr, **kwargs)
        self.plane = plane

    def forward(self, model, x, edge_index=None, node_index=None, **kwargs):
        graph = kwargs['graph']

        prediction = self._train(model, graph, node_index)

        node_mask = {key: self._post_process_mask(
                self.node_mask[key],
                self.hard_node_mask[key],
            ) for key in self.node_mask.keys()}

        edge_mask = {key: self._post_process_mask(
                self.edge_mask[key],
                self.hard_edge_mask[key],
            ) for key in self.edge_mask.keys()}

        self._clean_model(model)
        explainer = Explanation(node_mask=node_mask, edge_mask=edge_mask)

        for plane in self.plane: 
            explainer[plane] = {}
            explainer[plane]['pos'] = graph[plane]['pos']
            explainer[plane]['pred_label'] = prediction[plane]['x_semantic']
            explainer[plane]['sem_label'] = graph[plane]['y_semantic']
            explainer[plane]['x']=graph[plane]['x']

        return explainer
    
    def assign_planar_masks(self, graph, plane): 
        node_mask_type = self.explainer_config.node_mask_type

        x_mask = graph[plane]['x']
        edge_index_mask = graph[plane, 'plane', plane]['edge_index'].to(torch.float)

        (N, F), E = x_mask.size(), edge_index_mask.size(1) 
        node_mask = None 

        if node_mask_type == MaskType.object:
            node_mask = Parameter(torch.randn(N, 1) * 0.1)
        elif node_mask_type == MaskType.attributes:
            node_mask = Parameter(torch.randn(N, F) * 0.1)
        std = torch.nn.init.calculate_gain('relu') * math.sqrt(2.0 / (2 * N))
        edge_mask = Parameter(torch.randn(E) * std)

        parameters = []
        if node_mask is not None:
            parameters.append(node_mask)

        if edge_mask is not None:
            parameters.append(edge_mask)
        
        return parameters, node_mask, edge_mask

    def update_planar_graph(self, graph, plane): 
        update_graph = copy.copy(graph)
        x = graph[plane]['x']
        h = x if self.node_mask is None else x * self.node_mask.sigmoid()
        update_graph[plane]['x'] = h 

        # Update the edge masks 
        edges =  graph[plane, 'plane', plane]['edge_index'].to(torch.float)
        #edge_mask_h = edges if self.edge_mask is None else edges * self.edge_mask.sigmoid()
        update_graph[plane, 'plane', plane]['edge_index'] = self.filter_edges(edges, self.edge_mask)

        return update_graph

    def get_masked_subgraph(self, graph): 
          
        for plane in self.planes: 

            node_mask_value = self.node_mask[plane].sigmoid().ravel()
            topk_nodes = torch.topk(node_mask_value, k=int(len(node_mask_value)/3), dim=0)

            edge_weights = self.edge_mask[plane].sigmoid()
            tokp_edges = torch.topk(edge_weights.ravel(), k=int(len(edge_weights.ravel())/3), dim=0)

            graph[(plane, "plane", plane)]['edge_index'] = graph[(plane, "plane", plane)]['edge_index'][:,tokp_edges.indices]
            graph[(plane, "plane", plane)]['x'] = graph[plane]['x'][topk_nodes.indices]
        return graph 
    
    def filter_edges(self, edges, filter): 
        filter = (filter.sigmoid() >= 0.5)
        edges = edges[:,filter]
        return edges.to(int)

    def assign_nexus_masks(self, graph): 
        # Soft Masks
        self.node_mask = {}
        self.edge_mask = {}

        # Hard Masks 
        self.hard_node_mask = {}
        self.hard_edge_mask = {}
        all_parameters = []

        for plane in self.plane: 
            ## planar masks 
            parameters, node_mask, edge_mask = self.assign_planar_masks(graph, plane)
            all_parameters+=parameters
            self.node_mask[plane] = node_mask
            self.edge_mask[plane] = edge_mask

            self.hard_edge_mask[plane] = torch.ones_like(edge_mask).to(bool)
            self.hard_node_mask[plane] = torch.ones_like(node_mask).to(bool)

            ## nexus masks 
            nexus_edge = graph[(plane, 'nexus', 'sp')]['edge_index'].to(torch.float)
            N, _ = node_mask.size()
            E = nexus_edge.size(1)
            std = torch.nn.init.calculate_gain('relu') * math.sqrt(2.0 / (2 * N))
            edge_mask = Parameter(torch.randn(E) * std)

            self.edge_mask[f"{plane}_nexus"] = edge_mask
            self.hard_edge_mask[f"{plane}_nexus"] = torch.ones_like(edge_mask).to(bool)

            all_parameters.append(edge_mask)

        return all_parameters
    
    def update_nexus_masks(self, graph): 
        update_graph = copy.copy(graph)

        for plane in self.plane: 
            plane_node_mask = self.node_mask[plane]
            plane_edge_planar_mask = self.edge_mask[plane]
            plane_edge_nexus_mask = self.edge_mask[f"{plane}_nexus"]

            #Update the node mask 
            x = graph[plane]['x']
            h = x if plane_node_mask is None else x * plane_node_mask.sigmoid()
            update_graph[plane]['x'] = h 

            # Update the edge masks 
            edges =  graph[plane, 'plane', plane]['edge_index'].to(torch.float)            
            update_graph[plane, 'plane', plane]['edge_index'] = self.filter_edges(edges, plane_edge_planar_mask)

            # Update the nexus edge masks 
            edges =  graph[plane, 'nexus', 'sp']['edge_index'].to(torch.float)
            update_graph[plane, 'nexus', 'sp']['edge_index'] = self.filter_edges(edges, plane_edge_nexus_mask)

        return update_graph
    

    def _train(self, model, graph, node_index=None, **kwargs):
        graph.requires_grad=True

        parameters = self.assign_nexus_masks(graph)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
 
        for i in range(self.epochs):
            optimizer.zero_grad()

            stepped_graph = self.update_nexus_masks(graph)

            model.step(stepped_graph)
            
            y = torch.concat([
                stepped_graph[plane]["y_semantic"] 
                for plane in self.plane
                ]).to(torch.float)
            
            y_hat = torch.concat([
                torch.argmax(stepped_graph[plane]['x_semantic'], dim=-1).to(torch.float)
                for plane in self.plane
            ]).to(torch.float)

            assert y.size() == y_hat.size(), print(f"{y.size()} vs {y_hat.size()}")## Personal check that things are not weirdly transposed

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
                self.hard_node_mask = {
                    key: self.node_mask[key].grad!=0  
                    for key in self.node_mask.keys()
                    }
                
            if i == 0 and self.edge_mask is not None:
                self.hard_edge_mask = {
                    key: self.edge_mask[key].grad!=0 
                    for key in self.edge_mask.keys()
                    }
                
        return stepped_graph

    def get_nexus_m(self): 
        m_node = torch.concat([
            self.node_mask[key][self.hard_node_mask[key]].sigmoid() 
            for key in self.node_mask.keys()])
        m_edge = torch.concat([
            self.edge_mask[key][self.hard_edge_mask[key]].sigmoid() 
            for key in self.edge_mask.keys()])
        
        return m_node, m_edge


    def _loss(self, y_hat, y): 

        loss = self._loss_multiclass_classification(y_hat, y)
        m_node, m_edge = self.get_nexus_m()

        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m_edge)
        ent = -m_edge * torch.log(m_edge + self.coeffs['EPS']) - (
            1 - m_edge) * torch.log(1 - m_edge + self.coeffs['EPS'])
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_reduce(m_node)
        ent = -m_node * torch.log(m_node + self.coeffs['EPS']) - (
            1 - m_node) * torch.log(1 - m_node + self.coeffs['EPS'])
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss
