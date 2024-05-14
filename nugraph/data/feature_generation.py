import copy
import numpy as np
import torch

from torch_geometric.transforms import BaseTransform


class FeatureGeneration(BaseTransform):
    def __init__(
        self, n_original_features=4, distance_metric="euclidean", planes=("v", "u", "y")
    ) -> None:
        self.distance = {"euclidean": self.euclidean_distance}[distance_metric]
        self.planes = planes
        self.n_original_features = n_original_features

    def euclidean_distance(self, x, y):
        return np.linalg.norm(np.subtract(x, y))

    def _get_edges(self, graph):
        try:
            edges = graph.collect("edge_index")
        except AttributeError:
            graph = graph[0]
            edges = graph.collect("edge_index")
        return edges

    def _get_node_positions(self, graph):
        try:
            positions = graph.collect("pos")
        except AttributeError:
            graph = graph[0]
            positions = graph.collect("pos")
        return positions

    def edge_length(self, graph):
        positions = self._get_node_positions(graph)
        edges = self._get_edges(graph)

        edge_lengths = {}
        for plane in self.planes:
            position = positions[plane]
            plane_edges = edges[(plane, "plane", plane)]

            edge_positions = [
                (
                    position[plane_edges[0][edge_index]],
                    position[plane_edges[1][edge_index]],
                )
                for edge_index in range(len(plane_edges[0]))
            ]
            _length = torch.tensor(
                [self.distance(edge_1, edge_2) for edge_1, edge_2 in edge_positions]
            )
            edge_lengths[plane] = _length.unsqueeze(-1)
        return edge_lengths

    def edge_slope(self, graph):
        positions = self._get_node_positions(graph)
        edges = self._get_edges(graph)
        edge_slope = {}

        for plane in self.planes:
            position = positions[plane]
            plane_edges = edges[(plane, "plane", plane)]

            edge_positions = [
                (
                    position[plane_edges[0][edge_index]],
                    position[plane_edges[1][edge_index]],
                )
                for edge_index in range(len(plane_edges[0]))
            ]
            m = torch.tensor(
                [
                    self.distance(edge_1[1], edge_2[1])
                    / self.distance(edge_1[0], edge_2[0])
                    for edge_1, edge_2 in edge_positions
                ]
            )
            edge_slope[plane] = m.unsqueeze(-1)
        return edge_slope

    def node_slope(self, graph):
        positions = self._get_node_positions(graph)
        node_slope = {}
        for plane in self.planes:
            m = torch.tensor(positions[plane][:, 1] / positions[plane][:, 0])
            node_slope[plane] = m.unsqueeze(-1)

        return node_slope

    def __call__(self, graph):
        edge_slope = self.edge_slope(graph)
        edge_length = self.edge_length(graph)
        edge_features = [edge_slope, edge_length]  # Add new edge features here

        node_slope = self.node_slope(graph)
        node_features = [node_slope]  # Add new node features here

        add_graph = copy.deepcopy(graph)
        for plane in self.planes:
            _features = [feature[plane] for feature in edge_features]
            features = torch.concat(_features, dim=-1).T
            add_graph[plane, plane].features = torch.concat(features, dim=-1).T

            _features = [feature[plane] for feature in node_features]
            _features.append(graph.collect("x")[plane][:, : self.n_original_features])
            add_graph[plane]["x"] = torch.concat(_features, dim=-1)

        return add_graph
