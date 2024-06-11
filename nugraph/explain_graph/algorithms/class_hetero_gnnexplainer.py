from nugraph.explain_graph.algorithms.hetero_gnnexplainer import HeteroGNNExplainer

import copy
import torch
import numpy as np


class MultiEdgeHeteroGNNExplainer(HeteroGNNExplainer):
    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        plane="u",
        incorrect_only=False,
        **kwargs,
    ):
        super().__init__(epochs, lr, plane, **kwargs)
        self.loss_history = []
        self.incorrect_only = incorrect_only
        self.semantic_classes = ["MIP", "HIP", "shower", "michel", "diffuse"]

    def forward(self, model, graph, nodes):
        self.target_class = 0

        # Focus on the prediction for each graph
        # - the pruned subgraph should be the subgraph that allow a certain class to be predicted
        multiple_loss = []
        explainations = {}
        for class_index in range(len(self.semantic_classes)):
            training_graph = copy.deepcopy(graph)
            explainations[class_index] = super().forward(model, training_graph)

            self.target_class += 1
            multiple_loss.append(self.loss_history)
            self.loss_history = []

        self.loss_history = multiple_loss
        return explainations

    def plot_loss(self, file_name):
        import matplotlib.pyplot as plt

        plt.close("all")
        norm = np.mean(np.array(self.loss_history), axis=0)
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
            other=torch.tensor([1], device=self.device),
        ).to(dtype=torch.float)
        y = filter.to(torch.device(self.device))
        return y

    def _classification_loss(self, y_hat, y):
        binary_y_hat = self._make_binary(y_hat)
        binary_y = self._make_binary(y)

        return torch.nn.BCELoss()(binary_y_hat, binary_y)
