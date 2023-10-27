from typing import Any, Dict, Optional, Union
import torch
from torch import Tensor
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
    def __init__(self, epochs: int = 100, lr: float = 0.01, planes=[], **kwargs):
        super().__init__(epochs, lr, **kwargs)
        self.planes = planes

    def forward(self, model, x, edge_index, **kwargs):
        graph = kwargs['graph']

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

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)
    
    def _train(self, model, graph,  **kwargs):

        accelerator, device = util.configure_device()
        trainer = pl.Trainer(accelerator=accelerator,
                            logger=False)
        predictions = trainer.predict(model, dataloaders=graph)[0]

        dataset = graph.dataset
        for plane in self.planes: 
            ## Use only a single plane - the x tensor used for analysis is different than the tensor used for the forward prediction
            x_mask = dataset[plane]['x']
            edge_index_mask = dataset[plane, 'plane', plane]['edge_index']

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
                
                y_hat =  predictions[plane]['x_semantic']
                y = dataset[plane]['y_semantic']
                
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