import matplotlib.pyplot as plt
import networkx as nx

import torch_geometric as pyg
import torch
import pandas as pd
from nugraph.explain_graph.load import Load

class EdgeVisuals: 

    def __init__(self, test=False) -> None:
        load = Load(test=test)
        self.model = load.model
        self.data = load.data.dataset
        self.predictions = load.predictions

    def plot_truth_labels(self): 
        pass 

    def plot_model_labels(self): 
        pass

    def plot_plane(self, graph, plane='u'): 
        
        subgraph_nx = nx.Graph()
        # position = graph[plane]['pos'] 
        # x = graph[plane]['x'] 

        # for pos, x_val in zip(position, x):
        #     subgraph_nx.add_node(pos, x=x_val)

        edge_node_1 = graph[(plane, "plane", plane)]['edge_index'][0]
        edge_node_2 = graph[(plane, "plane", plane)]['edge_index'][1]
        for v, u in zip(edge_node_1, edge_node_2):
            subgraph_nx.add_edge(v, u)

        position = {node: graph[plane]['pos'][node] for node in range(len(graph[plane]['pos']))}

        true_labels = {node: graph[plane]['sem_label'] for node in range(len(graph[plane]['pos']))}
        pred_labels = {node: graph[plane]['pred_label'] for node in range(len(graph[plane]['pos']))}

        truth = nx.draw(subgraph_nx, pos=position, labels=true_labels) 
        prediction = nx.draw(subgraph_nx, pos=position, labels=pred_labels) 
        plt.savefig("test_plot.png")
        return truth, prediction

    def get_graph(self, data_index): 

        graph = self.data.get(data_index)
        prediction = self.predictions[data_index]

        for plane in ['u', 'v', 'y']: 
            plane_prediction = pd.Categorical(prediction['x_semantic'][plane]).codes
            graph[plane]['pred_label'] = plane_prediction
            graph[plane]['sem_label'] = pd.Categorical(prediction['y_semantic'][plane]).codes

        return graph 
    
    def plot(self, data_index=0): 

        graph = self.get_graph(data_index)

        fig, subplots = plt.subplots(2, 3)
        
        u_plane = self.plot_plane(graph, plane='u')
        v_plane = self.plot_plane(graph, plane='v')
        y_plane = self.plot_plane(graph, plane='y')

        subplots[0,1], subplots[0,2] = u_plane[0], u_plane[1]
        subplots[0,2].set_x_label("U Plane")
        subplots[0,1].set_y_label("Prediction")
        subplots[0,2].set_y_label("Truth")

        subplots[1,1], subplots[1,2] = v_plane[0], v_plane[1]
        subplots[1,2].set_x_label("V Plane")

        subplots[2,1], subplots[2,2] = y_plane[0], y_plane[1]
        subplots[2,2].set_x_label("y Plane")

        fig.supxlabel("wire")
        fig.supylabel("time")

        plt.savefig("test_plot.png")


if __name__=='__main__': 
    EdgeVisuals(test=True).plot()