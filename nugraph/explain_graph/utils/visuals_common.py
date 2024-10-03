import networkx as nx
import numpy as np
import torch

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt


def highlight_nodes(graph, node_list, plane, axes):
    x = [graph[plane]["pos"][node][0] for node in node_list]
    y = [graph[plane]["pos"][node][1] for node in node_list]
    axes.scatter(x, y, s=80, color="orange", marker=(5, 1))


def make_subgraph_kx(graph, plane, semantic_classes=None):
    subgraph_nx = nx.Graph()

    edge_node_1 = graph[(plane, "plane", plane)]["edge_index"][0]
    edge_node_2 = graph[(plane, "plane", plane)]["edge_index"][1]

    if "weight" in graph[(plane, "plane", plane)].keys():
        weight = graph[(plane, "plane", plane)]["weight"]
        for v, u, w in zip(edge_node_1, edge_node_2, weight):
            if w.sigmoid() != 0:
                subgraph_nx.add_edge(v, u, weight=w)

    else:
        for v, u in zip(edge_node_1, edge_node_2):
            subgraph_nx.add_edge(v, u)

    nodes = subgraph_nx.nodes

    try:
        position = {node: graph[plane]["pos"][node].tolist() for node in nodes}
    except IndexError:  # If the nodes are given as tensors
        position = {
            node: graph[plane]["pos"][int(node.item())].tolist() for node in nodes
        }

    nx.set_node_attributes(subgraph_nx, position, "pos")
    if semantic_classes is not None:
        true_labels = {
            node: semantic_classes[torch.argmax(graph[plane]["y_semantic"][node])]
            for node in nodes
        }
        return subgraph_nx, true_labels

    else:
        return subgraph_nx


def extract_plane_subgraph(graph, plane):
    subgraph = graph[plane]
    subgraph[(plane, "plane", plane)] = {
        "edge_index": graph[(plane, "plane", plane)]["edge_index"]
    }
    subgraph[(plane, "nexus", "sp")] = {
        "edge_index": graph[(plane, "nexus", "sp")]["edge_index"]
    }
    return subgraph


def extract_edge_weights(graph, plane, return_value=False, cmap="viridis", nexus=False):
    weights = [1 for _ in range(len(graph[(plane, "plane", plane)]))]
    from_graph = False

    weight_colors = "grey"

    # edge_attr is more acturate for picking mask shapes
    if not nexus:
        plane_name = plane
    else:
        plane_name = "sp"

    if hasattr(graph[plane, plane_name], "edge_mask"):
        weights = graph[plane, plane_name].edge_mask
        from_graph = True

    if from_graph:
        try:
            weights = (weights - weights.min()) / (weights.max() - weights.min())

            cNorm = colors.Normalize(vmin=0, vmax=weights.max())
            color_map = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))
            weight_colors = [color_map.to_rgba(weight) for weight in weights]

        except RuntimeError:
            pass

    if return_value:
        return weights

    return weight_colors


def extract_node_weights(
    graph, plane, nodes, node_field="node_mask", scale=True, color=False, cmap="viridis"
):
    if node_field in graph[plane].keys():
        node_size = np.array([graph[plane][node_field][node] for node in nodes])
        if scale:
            node_size = (
                (node_size - node_size.min()) / (node_size.max() - node_size.min()) * 30
            )
            node_size = node_size.ravel()
            if color:
                cNorm = colors.Normalize(vmin=0, vmax=node_size.max())
                color_map = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))
                weight_colors = [color_map.to_rgba(weight) for weight in node_size]
                return weight_colors

    else:
        node_size = [2 for _ in range(len(graph[plane]["x"]))]

    return node_size


def extract_class_subgraphs(graph, planes, class_index):
    labels = graph.collect("y_semantic")
    nodes = {}
    for plane in planes:
        nodes[plane] = labels[plane] == class_index

    subgraph = graph.subgraph(nodes)
    return subgraph
