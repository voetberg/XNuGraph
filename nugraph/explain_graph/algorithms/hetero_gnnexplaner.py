from typing import Optional, Union
import torch
from torch import Tensor
from torch_geometric.explain import Explanation, GNNExplainer 
from torch_geometric.explain.config import ModelMode
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.data import Batch


class HeteroGNNExplainer(GNNExplainer): 
    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__(epochs, lr, **kwargs)

    def forward(self, model, graph, plane='u', **kwargs) -> Explanation:
        self._train(model, graph, plane=plane, **kwargs)

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

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)
    

    def _train(self, model, graph, plane='u', **kwargs):
        ## Use only a single plane - the x tensor used for analysis is different than the tensor used for the forward prediction

        x_mask = graph[plane]['x']
        edge_index_mask = graph[plane, 'plane', plane].edge_index

        self._initialize_masks(x_mask, edge_index_mask)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index_mask, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            model.step(graph)

            y_hat, y = graph[plane]['x_semantic'], graph[plane]['y_semantic']
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