import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

import torch_geometric as pyg
import torch
import pandas as pd 
from nugraph.explain_graph.load import Load

import matplotlib.colors as colors
import matplotlib.cm as cmx


class EdgeVisuals: 

    def __init__(self, 
                test=False, 
                checkpoint_path=None, 
                data_path=None, 
                planes=['u', 'v', 'y'], 
                semantic_classes = ['MIP','HIP','shower','michel','diffuse'],
                weight_colormap='viridis') -> None:
        if (data_path is not None) or (test): 
            load = Load(test=test, checkpoint_path=checkpoint_path, data_path=data_path, batch_size=1)
            self.model = load.model
            self.data = load.data.dataset
            self.predictions = load.predictions

        self.semantic_classes = semantic_classes
        self.planes = planes
        self.cmap = weight_colormap

    def extract_weights(self, graph, plane, return_value=False): 
        if "weight" in graph[(plane, "plane", plane)].keys(): 
            weights = graph[(plane, "plane", plane)]['weight']

            weights = (weights - weights.min())/(weights.max() - weights.min())
            cNorm  = colors.Normalize(vmin=0, vmax=weights.max())
            color_map = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(self.cmap))
            weight_colors = [color_map.to_rgba(weight) for weight in weights]
        else: 
            weights = [1 for _ in range(len(graph[(plane, "plane", plane)]))]
            weight_colors = 'grey'

        if "node_mask" in graph[plane].keys(): 
            node_size = graph[plane]["node_mask"]
            node_size = (node_size - node_size.min())/(node_size.max() - node_size.min())*20
            node_size = node_size.ravel().tolist()

        else: 
            node_size = [5 for _ in range(len(graph[plane]['x']))]

        if return_value: 
            return weights
        
        else: 
            return weight_colors, node_size
    
    def plot_graph(self, graph, subgraph, plane, node_list, axes): 
        nodes = subgraph.nodes

        position = {node: graph[plane]['pos'][node].tolist() for node in nodes}
        if node_list is None: 
            node_list = list(subgraph)

        weight_colors, node_size = self.extract_weights(graph, plane)
        node_size = [node_size[int(node.item())] for node in nodes]

        edge_list = subgraph.edges(node_list)
        drawn_plot = nx.draw_networkx(
            subgraph,
              pos=position, 
              with_labels=False,  
              nodelist=node_list, 
              edgelist=edge_list, 
              node_size=node_size, 
              width=5, 
              node_color='red',
              edge_color=weight_colors,
              ax=axes) 
        return drawn_plot
 
    def make_subgraph(self, graph, plane='u'): 
        
        subgraph_nx = nx.Graph()

        edge_node_1 = graph[(plane, "plane", plane)]['edge_index'][0]
        edge_node_2 = graph[(plane, "plane", plane)]['edge_index'][1]

        if "weight" in graph[(plane, "plane", plane)].keys(): 
            weight = graph[(plane, "plane", plane)]['weight']
            for v, u, w in zip(edge_node_1, edge_node_2, weight):
                if w.sigmoid()!=0: 
                    subgraph_nx.add_edge(v, u, weight=w)

        else: 
            for v, u in zip(edge_node_1, edge_node_2):
                subgraph_nx.add_edge(v, u)

        nodes = subgraph_nx.nodes
        position = {node: graph[plane]['pos'][node].tolist() for node in nodes}

        nx.set_node_attributes(subgraph_nx, position, 'pos')

        true_labels = {node: self.semantic_classes[torch.argmax(graph[plane]['sem_label'][node])] for node in nodes}
        pred_labels = {node: self.semantic_classes[torch.argmax(graph[plane]['pred_label'][node])] for node in nodes}

        return subgraph_nx, true_labels, pred_labels
    
    def get_graph(self, data_index): 

        graph = self.data.get(data_index)
        prediction = self.predictions[data_index]
        for plane in self.planes: 
            graph[plane]['pred_label'] = prediction[plane]['x_semantic']
            graph[plane]['sem_label'] = prediction[plane]['y_semantic']

        return graph 
    
    def plot(self, data_index=0, graph=None, incorrect_items=False, semantic_class=None, not_in=False, title="", outdir=".", file_name="prediction_plot.png"): 
        """_summary_

        Args:
            data_index (int, optional): _description_. Defaults to 0.
            graph (_type_, optional): _description_. Defaults to None.
            incorrect_items (bool, optional): _description_. Defaults to True.
            semantic_class (_type_, optional): _description_. Defaults to None.
            not_in (bool, optional): _description_. Defaults to False.
            title (str, optional): _description_. Defaults to "".
            outdir (str, optional): _description_. Defaults to ".".
            file_name (str, optional): _description_. Defaults to "prediction_plot.png".
        """
        figure, subplots = plt.subplots(1, 3, figsize=( 16*3, 16))

        for plane, subplot in zip(self.planes, subplots): 

            if graph is None: 
                graph = self.get_graph(data_index)

            subgraph, labels, predictions = self.make_subgraph(graph, plane=plane)

            node_list = subgraph.nodes 
            if incorrect_items: 
                node_list = [node for node in node_list if predictions[node]!=labels[node]]

            if semantic_class is not None: 
                assert semantic_class in self.semantic_classes
                if not_in: 
                    node_list = [node for node in node_list if labels[node]!=semantic_class]
                else: 
                    node_list = [node for node in node_list if labels[node]==semantic_class]


            self.plot_graph(graph, subgraph, plane, node_list, subplot)
            subplot.set_title(plane)

        figure.supxlabel("wire")
        figure.supylabel("time")
        figure.suptitle(title)

        plt.colorbar(
            cmx.ScalarMappable(
                norm=colors.Normalize(vmin=0, vmax=1), 
                cmap=plt.get_cmap(self.cmap)), 
            ax=subplots.tolist())

        plt.savefig(f"{outdir.rstrip('/')}/{file_name}")


    def _interative_edge(self, subgraph, edge_weights): 
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = subgraph.nodes[edge[0]]['pos']
            x1, y1 = subgraph.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)


        colors = [f'rgb({c[0]},{c[1]},{c[2]})' for c in edge_weights]
        edge_traces = go.Scatter(x=edge_x, y=edge_y,
            mode='lines', 
            name='Edges',
            hoverinfo='none',
            line=dict(
                color='cornflowerblue',
            ),
            line_width=2,
            )
        # for c in enumerate(colors): 
        #     edge_traces.
        
        return edge_traces
        
    def _interactive_nodes(self, subgraph, label, class_label='0'): 
        node_x = []
        node_y = []

        for node in subgraph.nodes():
            x, y = subgraph.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        size = (label - label.min())/(label.max() - label.min())*10
        size = size.tolist()
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text', 
            name=class_label,
            marker=dict(
                showscale=True,
                color="black",
                size=size)
        )

        node_trace.text = label.tolist()

        return node_trace
    

    def _grouped_interative_nodes(self, subgraph, label, x): 
        labels = pd.DataFrame(label.values(),index=label.keys())
        labels.columns = ['label']
        traces = []
        for unique in labels["label"].unique(): 
            nodes_to_include = labels[labels['label']==unique].index.tolist()
            label_group = subgraph.__class__()
            label_group.add_nodes_from((n, subgraph.nodes[n]) for n in nodes_to_include)
            x_values = torch.Tensor([x[i.item()] for i in nodes_to_include])
            traces.append(self._interactive_nodes(label_group, label=x_values, class_label=unique))
        return traces

    def interactive_plot(self, data_index=0, graph=None, plane="u", node_label_field="label", x_index=0, group_labels=False, outdir=".", file_name="prediction_plot.html"): 
        """
        Produce an interactive plot where the nodes are click-able and the field can be moved.
        Can only plot one field at a time, saves each result to an html. 
        Args:
            plane (str, optional): Field to plot. Defaults to u.

        """
        if graph is None: 
            graph = self.get_graph(data_index)

        subgraph, labels, predictions = self.make_subgraph(graph, plane=plane)
        

        node_labels = {
            "label":labels, 
            "predictions": predictions, 
        }[node_label_field]

        nodes = self._grouped_interative_nodes(subgraph, node_labels, graph[plane]['x'][:,x_index].tolist())

        weights, _ = self.extract_weights(graph, plane)
        edges = self._interative_edge(subgraph, weights)

        # Flatten the possible mul;tiple nodes
        data = [edges]
        for node in nodes: 
            data.append(node)

        fig = go.Figure(
            data=data, 
            layout=go.Layout(showlegend=False)
            
        )
        fig.write_html(f"{outdir.rstrip('/')}/{file_name}.html")


    def class_seperated_plot(self, data_index=0, graph=None, plane="u", node_label_field="label", outdir=".", file_name="prediction_plot", interactive=False): 
        if interactive: 
            self.interactive_plot(data_index, graph, plane=plane, group_labels=True, outdir=outdir, file_name=file_name)
        else: 
            ""
        


if __name__=='__main__': 
    vis = EdgeVisuals(test=True)
    vis.plot(incorrect_items=False)
    #vis.interactive_plot(group_labels=True)