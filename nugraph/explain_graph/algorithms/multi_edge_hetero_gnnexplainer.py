from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroGNNExplainer
from nugraph.explain_graph.utils.masking_utils import apply_predefined_mask
from torch_geometric.explain import Explanation

import copy 
import torch

class MultiEdgeHeteroGNNExplainer(HeteroGNNExplainer): 
    def __init__(self, epochs: int = 100, lr: float = 0.01, plane='u', **kwargs):
        super().__init__(epochs, lr, plane, **kwargs)
        self.loss_history = {}

    def _classwise_subgraphs(self, graph, predictions): 
        unique_category = torch.unique(torch.concat([predictions[plane] for plane in self.planes]))
        subgraphs = {}
        for prediction_class in unique_category: 
            node_mask = {}
            edge_mask = {}
            nexus_mask = {}

            for plane in self.planes: 
                plane_node_mask = torch.argmax(graph[plane]['x_semantic'], axis=-1) == prediction_class
                nodes_include = plane_node_mask.nonzero().squeeze()

                plane_edges = graph[(plane, "plane", plane)]['edge_index']
                edge_mask[plane] =torch.bitwise_and(torch.isin(plane_edges[0], nodes_include), torch.isin(plane_edges[1], nodes_include))
                nexus_edges =  graph[(plane, 'nexus', 'sp')]['edge_index']
                nexus_mask[plane] = torch.bitwise_and(torch.isin(nexus_edges[0], nodes_include), torch.isin(nexus_edges[1], nodes_include))
                node_mask[plane] = plane_node_mask

            subgraphs[prediction_class.item()] = apply_predefined_mask(graph.detach(), node_mask, edge_mask, nexus_mask, self.planes)
        return subgraphs


    def plot_loss(self, file_name):
        return super().plot_loss(file_name)
    
    def forward(self, model, graph):
        prediction = copy.deepcopy(graph)
        model.step(prediction)

        predicted_classes = {plane: torch.argmax(prediction[plane]['x_semantic'], axis=-1) for plane in self.planes}
        classwise_subgraphs = self._classwise_subgraphs(prediction, predicted_classes)
        explainations = {}
        for item, graph in classwise_subgraphs.items(): 
            _, history = self._train(model, graph, loss_history=[])
            self.loss_history[item] = history

            node_mask = {key: self._post_process_mask(
                    self.node_mask[key],
                    self.hard_node_mask[key],
                ) for key in self.node_mask.keys()}

            edge_mask = {key: self._post_process_mask(
                    self.edge_mask[key],
                    self.hard_edge_mask[key],
                ) for key in self.edge_mask.keys()}

            self._clean_model(model)
            explainations[item] = Explanation(node_mask=node_mask, edge_mask=edge_mask, kwargs={"graph":graph})
  
        return explainations

    
    def plot_loss(self, file_name):
        import matplotlib.pyplot as plt 
        plt.close('all')

        for key in self.loss_history: 
            norm = torch.nn.functional.normalize(torch.Tensor(self.loss_history[key]), dim=0)
            plt.plot(range(len(norm)), norm, label=key)

        plt.legend()
        plt.xlabel("Iteration") 
        plt.ylabel("Normalized Entropy Loss")

        plt.savefig(file_name)
        plt.close("all")