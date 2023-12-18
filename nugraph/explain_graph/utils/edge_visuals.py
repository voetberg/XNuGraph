import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

import torch
import pandas as pd 
import math 

import matplotlib.colors as colors
import matplotlib.cm as cmx

# Common 

def make_subgraph_kx(graph, plane, semantic_classes=None): 
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
    if semantic_classes is not None: 
        true_labels = {node: semantic_classes[torch.argmax(graph[plane]['y_semantic'][node])] for node in nodes} 
        return subgraph_nx, true_labels

    else: 
        return subgraph_nx
    
def extract_edge_weights(graph, plane, return_value=False, cmap='viridis'): 

    weights = [1 for _ in range(len(graph[(plane, "plane", plane)]))]
    weight_colors = 'grey'

    if "edge_mask" in graph.keys: 
        weights = graph["edge_mask"][plane]
        if weights.numel() != 0: 
            weights = (weights - weights.min())/(weights.max() - weights.min())

            cNorm  = colors.Normalize(vmin=0, vmax=weights.max())
            color_map = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))
            weight_colors = [color_map.to_rgba(weight) for weight in weights]

    
    if return_value: 
        return weights
    return weight_colors

def extract_node_weights(graph, plane, node_field='node_mask', scale=True): 
    if node_field in graph[plane].keys(): 
        node_size = graph[plane][node_field]
        if scale: 
            node_size = (node_size - node_size.min())/(node_size.max() - node_size.min())*20
            node_size = node_size.ravel().tolist()

    else: 
        node_size = [2 for _ in range(len(graph[plane]['x']))]

    return node_size

class EdgeVisuals: 

    def __init__(self, 
                planes=['u', 'v', 'y'], 
                semantic_classes = ['MIP','HIP','shower','michel','diffuse'],
                weight_colormap='viridis') -> None:

        self.semantic_classes = semantic_classes
        self.planes = planes

        self.cmap = weight_colormap
    
    def plot_graph(self, graph, subgraph, plane, node_list, axes): 
        nodes = subgraph.nodes

        position = {node: graph[plane]['pos'][node].tolist() for node in nodes}
        if node_list is None: 
            node_list = list(subgraph)

        weight_colors = extract_edge_weights(graph, plane)

        edge_list = subgraph.edges(node_list)
        drawn_plot = nx.draw_networkx(
            subgraph,
              pos=position, 
              with_labels=False,  
              nodelist=node_list, 
              edgelist=edge_list, 
              width=5, 
              node_color='black',
              node_size=3,
              edge_color=weight_colors,
              ax=axes) 
        return drawn_plot
    
    def plot_nexus_weights(self, graph, plot): 
        """Plot a histogram of the nexus edge weights
        """
        assert "edge_mask" in graph.keys
        for plane in self.planes: 
            assert f"{plane}_nexus" in graph['edge_mask'].keys()
            nexus_edges = graph['edge_mask'][f'{plane}_nexus'].ravel() 
            
            bins = math.ceil(math.sqrt(len(nexus_edges)))
            bins = bins if bins!=0 else 10
            plot.hist(nexus_edges, alpha=0.6, label=plane, bins=bins)

        plot.set_title("Nexus Weight Importance")
        plot.legend()


    def plot(self, graph=None, title="", outdir=".", file_name="prediction_plot.png", nexus_distribution=False): 
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
        n_xplot = len(self.planes) + int(nexus_distribution)

        figure, subplots = plt.subplots(1, n_xplot, figsize=( 10*n_xplot, 10))#, sharex=True, sharey=True)

        for plane, subplot in zip(self.planes, subplots): 

            subgraph = make_subgraph_kx(graph, plane=plane)
            node_list = subgraph.nodes 

            self.plot_graph(graph, subgraph, plane, node_list, subplot)
            subplot.set_title(plane)

        if nexus_distribution: 
            self.plot_nexus_weights(graph, subplots[-1])


        figure.supxlabel("wire")
        figure.supylabel("time")
        figure.suptitle(title)

        plt.colorbar(
            cmx.ScalarMappable(
                norm=colors.Normalize(vmin=0, vmax=1), 
                cmap=plt.get_cmap(self.cmap)), 
            ax=subplots.tolist())

        plt.savefig(f"{outdir.rstrip('/')}/{file_name}")

        
class InteractiveEdgeVisuals: 
    def __init__(self, 
                 plane, 
                 semantic_classes = ['MIP','HIP','shower','michel','diffuse'], 
                 features=["1","2","3", "4", "5", "6"], 
                 feature_importance = False,
                 ) -> None:
        self.plane = plane 
        self.semantic_classes = semantic_classes
        self.features = features
        self.node_label_field = "node_mask" if feature_importance else "x"

    def _interative_edge(self, subgraph): 
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


        edge_traces = go.Scatter(x=edge_x, y=edge_y,
            mode='lines', 
            name='Edges',
            hoverinfo='none',
            line=dict(
                color='cornflowerblue',
            ),
            line_width=2,
            )
        
        return edge_traces
        
    def _interactive_nodes(self, subgraph, label:list[dict], class_label='0'): 
        node_x = []
        node_y = []

        for node in subgraph.nodes():
            x, y = subgraph.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        trace_text = [
            f"{self.node_label_field}<br>" + "<br>".join([f"{feature}:{round(point_label[feature], 5)}" for feature in point_label]) 
            for point_label in label
            ]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text', 
            name=class_label,
            text=trace_text,
            marker=dict(
                showscale=True,
                color="black")
        )


        return node_trace
    

    def _grouped_interative_nodes(self, subgraph, label, node_label): 
        labels = pd.DataFrame(label.values(),index=label.keys())
        labels.columns = ['label']
        traces = []
        for unique in labels["label"].unique(): 
            nodes_to_include = labels[labels['label']==unique].index.tolist()

            label_group = subgraph.__class__()
            label_group.add_nodes_from((n, subgraph.nodes[n]) for n in nodes_to_include)
            
            node_label = [node_label[node] for node in nodes_to_include]

            traces.append(self._interactive_nodes(label_group, label=node_label, class_label=unique))
        
        return traces

    def plot(self, graph, outdir=".", file_name="prediction_plot"): 
        """
        Produce an interactive plot where the nodes are click-able and the field can be moved.
        Can only plot one field at a time, saves each result to an html. 
        Args:
            plane (str, optional): Field to plot. Defaults to u.

        """

        subgraph, true_labels = make_subgraph_kx(
            graph, 
            plane=self.plane, 
            semantic_classes=self.semantic_classes
            )

        node_label = extract_node_weights(graph, plane=self.plane, node_field=self.node_label_field, scale=False)
        node_label = [
            {key: node[index].item() for index, key in enumerate(self.features)} 
            for node in node_label
            ]

        nodes = self._grouped_interative_nodes(subgraph, true_labels, node_label)
        edges = self._interative_edge(subgraph)

        # Flatten the possible mul;tiple nodes
        data = [edges]
        for node in nodes: 
            data.append(node)

        fig = go.Figure(
            data=data, 
            layout=go.Layout(showlegend=True)
            
        )
        fig.write_html(f"{outdir.rstrip('/')}/{file_name}.html")


if __name__=='__main__': 
    vis = EdgeVisuals(test=True)
    vis.plot(incorrect_items=False)


