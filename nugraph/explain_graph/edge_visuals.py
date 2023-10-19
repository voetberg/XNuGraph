import matplotlib.pyplot as plt
import networkx as nx

import torch_geometric as pyg
import torch
from nugraph.explain_graph.load import Load

class EdgeVisuals: 

    def __init__(self) -> None:
        load = Load()
        self.model = load.model
        self.data = load.data.dataset

    def weight_edges(self): 
        pass 

    def plot_truth_labels(self): 
        pass 

    def plot_plane(self, graph, plane='u'): 
        subgraph = graph.node_type_subgraph(node_types=[plane])
        nx.draw(subgraph, pos=subgraph.pos)

    def plot(self, data_index=0): 
        graph = self.data.get(data_index)

        del graph['metadata']
        del graph['evt']

        nx_graph = pyg.utils.to_networkx(graph)
        self.plot_plane(nx_graph, plane='u')


if __name__=='__main__': 
    EdgeVisuals().plot()