from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroGNNExplainer
from torch_geometric.explain import Explanation

import copy 
import torch

class MultiEdgeHeteroGNNExplainer(HeteroGNNExplainer): 
    def __init__(self, epochs: int = 100, lr: float = 0.01, plane='u', incorrect_only=False, **kwargs):
        super().__init__(epochs, lr, plane, **kwargs)
        self.loss_history = {}
        self.incorrect_only = incorrect_only
        self.semantic_classes = ['MIP','HIP','shower','michel','diffuse']

    def forward(self, model, graph):
        self.target_class = 0 

        # Focus on the prediction for each graph 
        # - the pruned subgraph should be the subgraph that allow a certian class to be predicted

        explainations = {}
        for class_index in range(len(self.semantic_classes)): 
            training_graph = copy.deepcopy(graph)
            _, history = self._train(model, training_graph, loss_history=[])
            self.loss_history = history

            node_mask = {key: self._post_process_mask(
                    self.node_mask[key],
                    self.hard_node_mask[key],
                ) for key in self.node_mask.keys()}

            edge_mask = {key: self._post_process_mask(
                    self.edge_mask[key], 
                    self.hard_edge_mask[key],
                ) for key in self.edge_mask.keys()}

            self._clean_model(model)
            for plane in self.planes: 
                training_graph[plane, plane].edge_mask = edge_mask[(plane, 'plane', plane)]
                training_graph[plane, "sp"].edge_mask = edge_mask[(plane, 'nexus', "sp")]
                training_graph[plane].node_mask = node_mask[plane]
                explainations[class_index] = training_graph
    
            self.target_class += 1

        return explainations

    
    def plot_loss(self, file_name):
        import matplotlib.pyplot as plt 
        plt.close('all')

        norm = torch.nn.functional.normalize(torch.Tensor(self.loss_history), dim=0)
        plt.plot(range(len(norm)), norm)

        plt.legend()
        plt.xlabel("Iteration") 
        plt.ylabel("Normalized Entropy Loss")

        plt.savefig(file_name)
        plt.close("all")

    def _make_binary(self, y): 
        filter = torch.where(
            torch.tensor([True], device=self.device), 
            torch.isin(y, torch.tensor([self.target_class], device=self.device)), 
            other=torch.tensor([1], device=self.device)).to(dtype=torch.float)
        y = filter.to(torch.device(self.device))
        return y


    def _classification_loss(self, y_hat, y):
        binary_y_hat = self._make_binary(y_hat)
        binary_y = self._make_binary(y)
        
        return torch.nn.BCELoss()(binary_y_hat, binary_y)