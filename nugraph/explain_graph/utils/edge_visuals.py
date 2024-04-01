from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import plotly.graph_objects as go

import torch
import pandas as pd
import math

import matplotlib.colors as colors
import matplotlib.cm as cmx

from nugraph.explain_graph.utils.visuals_common import (
    make_subgraph_kx,
    extract_class_subgraphs,
    extract_edge_weights,
    extract_node_weights,
    highlight_nodes,
)


class EdgeVisuals:
    def __init__(
        self,
        planes=["u", "v", "y"],
        semantic_classes=["MIP", "HIP", "shower", "michel", "diffuse"],
        weight_colormap="viridis",
    ) -> None:
        self.semantic_classes = semantic_classes
        self.planes = planes

        self.cmap = weight_colormap

    def plot_graph(self, graph, subgraph, plane, axes):
        nodes = subgraph.nodes
        position = {node: graph[plane]["pos"][node].tolist() for node in nodes}
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
            node_color="black",
            node_size=2,
            edge_color=weight_colors,
            ax=axes,
        )

        return drawn_plot

    def draw_ghost_plot(self, graph, plane, axes):
        subgraph = make_subgraph_kx(graph, plane=plane)
        position = {node: graph[plane]["pos"][node].tolist() for node in subgraph.nodes}

        drawn_plot = nx.draw_networkx(
            subgraph,
            pos=position,
            with_labels=False,
            width=1,
            node_color="grey",
            node_size=1,
            edge_color="grey",
            alpha=0.2,
            ax=axes,
        )

        return drawn_plot

    def plot_nexus_weights(self, graph, plot):
        """Plot a histogram of the nexus edge weights"""
        edge_weight = nexus_edges = graph.collect("edge_mask")
        for plane in self.planes:
            nexus_edges = edge_weight[(plane, "nexus", "sp")]

            bins = math.ceil(math.sqrt(len(nexus_edges)))
            bins = bins if bins != 0 else 10
            plot.hist(nexus_edges, alpha=0.6, label=plane, bins=bins)

        plot.set_title("Nexus Weight Importance")
        plot.legend()

    def _single_plot(
        self, graph, ghost_plot, subplots, nexus_distribution, key_hits=None
    ):
        if ghost_plot is None:
            ghost_plot = graph

        for plane, subplot in zip(self.planes, subplots):
            self.draw_ghost_plot(ghost_plot, plane, axes=subplot)
            subgraph = make_subgraph_kx(graph, plane=plane)
            self.plot_graph(graph, subgraph, plane, subplot)

            if key_hits is not None:
                highlight_nodes(
                    graph=graph, node_list=key_hits[plane], plane=plane, axes=subplot
                )

            subplot.set_title(plane)

        if nexus_distribution:
            self.plot_nexus_weights(graph, subplots[-1])

        plt.colorbar(
            cmx.ScalarMappable(
                norm=colors.Normalize(vmin=0, vmax=1), cmap=plt.get_cmap(self.cmap)
            ),
            ax=subplots.tolist(),
        )

    def plot(
        self,
        graph=None,
        title="",
        outdir=".",
        file_name="prediction_plot.png",
        ghost_plot=None,
        nexus_distribution=False,
        class_plot=False,
        key_hits=None,
    ):
        n_xplot = len(self.planes) + int(nexus_distribution)
        n_yplot = 1 if not class_plot else len(self.semantic_classes)
        figure, subplots = plt.subplots(
            n_yplot, n_xplot, figsize=(6 * n_xplot, 8 * n_yplot), sharex="col"
        )
        if ghost_plot is None:
            ghost_plot = graph

        if class_plot:
            self._plot_classes(graph, ghost_plot, subplots, key_hits)
        else:
            self._single_plot(graph, ghost_plot, subplots, nexus_distribution, key_hits)

        figure.supxlabel("wire")
        figure.supylabel("time")
        figure.suptitle(title)

        plt.savefig(f"{outdir.rstrip('/')}/{file_name}")

    def _plot_classes(self, graph, ghost_plot, subplots, key_hits=None):
        if ghost_plot is None:
            ghost_plot = deepcopy(graph)
        if not isinstance(graph, list):
            graph = [
                extract_class_subgraphs(graph, self.planes, class_index)
                for class_index in range(len(self.semantic_classes))
            ]

        for class_index, class_label in enumerate(self.semantic_classes):
            subplot_row = subplots[class_index, :]
            subplot_row[0].set_ylabel(class_label)
            if key_hits is not None:
                class_hits = key_hits[class_label]
            else:
                class_hits = None
            try:
                class_graph = graph[class_index]
                self._single_plot(
                    class_graph,
                    ghost_plot,
                    subplot_row,
                    nexus_distribution=False,
                    key_hits=class_hits,
                )
            except IndexError:
                pass

    def event_plot(self, graph, outdir, file_name="event_display.png", title=""):
        n_yplot = 2
        n_xplot = len(self.planes)
        figure, subplots = plt.subplots(
            n_yplot, n_xplot, figsize=(6 * n_xplot, 8 * n_yplot)
        )
        colors = [
            "grey",
            "darkorange",
            "dodgerblue",
            "limegreen",
            "palevioletred",
            "indigo",
        ]
        color_map = {index: color for index, color in zip([-1, 0, 1, 2, 3, 4], colors)}

        handles = [
            mpatches.Patch(color=color, label=label)
            for label, color in color_map.items()
        ]

        for plane, subplot in zip(self.planes, subplots.T):
            self.draw_ghost_plot(graph, plane, subplot[0])
            self.draw_ghost_plot(graph, plane, subplot[1])

            x, y = graph.collect("pos")[plane][:, 0], graph.collect("pos")[plane][:, 1]
            true_labels = graph.collect("y_semantic")[plane]
            predict_label = torch.argmax(graph.collect("x_semantic")[plane], axis=1)

            subplot[0].scatter(
                x, y, c=[color_map[label.item()] for label in true_labels]
            )
            subplot[1].scatter(
                x, y, c=[color_map[label.item()] for label in predict_label]
            )

        subplots[0][0].set_ylabel("Truth")
        subplots[1][0].set_ylabel("Prediction")
        subplots[0][-1].legend(handles=handles)
        subplots[1][-1].legend(handles=handles)

        figure.supxlabel("wire")
        figure.supylabel("time")
        figure.suptitle(title)

        plt.savefig(f"{outdir.rstrip('/')}/{file_name}")


class EdgeLengthDistribution:
    def __init__(
        self,
        out_path=".",
        include_nexus=True,
        planes=["u", "v", "y"],
        semantic_classes=["MIP", "HIP", "shower", "michel", "diffuse"],
        percentile=0.3,
    ) -> None:
        self.out_path = out_path
        self.include_nexus = include_nexus

        self.planes = planes
        self.semantic_classes = semantic_classes
        self.percentile = percentile

    def distance(self, x, y):
        return np.linalg.norm(np.subtract(x, y))

    def _extract_length(self, graph):
        try:
            positions = graph.collect("pos")
            edges = graph.collect("edge_index")
        except AttributeError:
            graph = graph[0]
            positions = graph.collect("pos")
            edges = graph.collect("edge_index")

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

            edge_lengths[plane] = torch.tensor(
                [self.distance(edge_1, edge_2) for edge_1, edge_2 in edge_positions]
            )
        return edge_lengths

    def plot(
        self,
        graph,
        style="scatter",
        split="class",
        file_name="plot.png",
    ):
        """ """
        n_rows = {"class": 1, "plane": 1, "all": len(self.planes), "none": 1}[split]
        n_cols = {
            "class": len(self.semantic_classes),
            "plane": len(self.planes),
            "all": len(self.semantic_classes),
            "none": 1,
        }[split]

        figure, subplots = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 6 * n_rows), sharex=True
        )

        {
            "class": self._plot_classwise,
            "plane": self._plot_planewise,
            "all": self._plot_all_seperate,
            "none": self._plot_all_together,
        }[split](graph, style, subplots)

        figure.suptitle(split)
        if style == "scatter":
            figure.supylabel("Importance")
            figure.supxlabel("Edge Length")
        elif style == "histogram":
            figure.supxlabel("Importance")

        plt.savefig(f"{self.out_path.rstrip('/')}/{file_name}")

    def _plot_classwise(self, graph, style, subplots, non_trained=False):
        if non_trained:
            graph = [
                extract_class_subgraphs(graph, self.planes, class_index)
                for class_index in range(len(self.semantic_classes))
            ]

        for label_index, label in enumerate(self.semantic_classes):
            try:
                subgraph = graph[label_index]
                importances = np.concatenate(
                    [
                        extract_edge_weights(subgraph, plane, return_value=True)
                        for plane in self.planes
                    ],
                    axis=0,
                )
                distances = self._extract_length(subgraph)
                distances = np.concatenate(
                    [distances[plane] for plane in self.planes], axis=0
                )

                nexus = None
                if self.include_nexus:
                    nexus = np.concatenate(
                        [
                            extract_edge_weights(
                                graph, plane, return_value=True, nexus=True
                            )
                            for plane in self.planes
                        ]
                    )
                self._single_plot(
                    style, subplots[label_index], distances, importances, nexus
                )

                subplots[label_index].set_xlabel(label)
            except IndexError:
                pass

    def _plot_planewise(self, graph, style, subplots):
        distances = self._extract_length(graph)

        for plane_index, plane in enumerate(self.planes):
            importance = np.concatenate(
                [extract_edge_weights(g, plane, return_value=True) for g in graph],
                axis=0,
            )
            distance = distances[plane]

            nexus = None
            if self.include_nexus:
                nexus = np.concatenate(
                    [
                        extract_edge_weights(
                            graph, plane, return_value=True, nexus=True
                        )
                        for plane in self.planes
                    ]
                )
            self._single_plot(style, subplots[plane_index], distance, importance, nexus)
            subplots[plane_index].set_title(plane)

    def _plot_all_together(self, graph, style, subplots):
        importances = np.concatenate(
            [
                extract_edge_weights(graph, plane, return_value=True)
                for plane in self.planes
            ]
        )
        distances = self._extract_length(graph)
        nexus = None
        if self.include_nexus:
            nexus = np.concatenate(
                [
                    extract_edge_weights(graph, plane, return_value=True, nexus=True)
                    for plane in self.planes
                ]
            )
        self._single_plot(style, subplots, distances, importances, nexus)

    def _plot_all_seperate(self, graph, style, subplots, non_trained=False):
        if non_trained:
            graph = [
                extract_class_subgraphs(graph, self.planes, class_index)
                for class_index in range(len(self.semantic_classes))
            ]

        for label_index, label in enumerate(self.semantic_classes):
            try:
                subgraph = graph[label_index]
                subplot_row = subplots[:, label_index]
                subplot_row[0].set_title(label)

                if not non_trained:
                    subgraph = extract_class_subgraphs(
                        subgraph, self.planes, label_index
                    )

                distances = self._extract_length(subgraph)

                for plane_index, plane in enumerate(self.planes):
                    importance = extract_edge_weights(
                        subgraph, plane, return_value=True
                    )
                    distance = distances[plane]

                    subplot_row[plane_index].set_ylabel(plane)
                    nexus = None
                    if self.include_nexus:
                        nexus = extract_edge_weights(
                            subgraph, plane, return_value=True, nexus=True
                        )
                    self._single_plot(
                        style=style,
                        subplot=subplot_row[plane_index],
                        distances=distance,
                        importances=importance,
                        nexus_importances=nexus,
                    )
            except IndexError:
                pass

    def _linreg_fit(self, x, y):
        from scipy.stats import linregress

        regression = linregress(x, y)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = regression.slope * x_fit + regression.intercept

        return x_fit, y_fit, regression.slope, regression.intercept, regression.rvalue

    def _scatter_plot(self, data, subplot):
        x = data[0]
        y = data[1]
        x_fit, y_fit, _, _, r = self._linreg_fit(x, y)

        subplot.scatter(x_fit, y_fit, alpha=0.6, label=f" R^2={round(r,4)}")
        subplot.scatter(x, y)
        subplot.legend()

    def _histogram(self, data, subplot):
        importances = data[1]
        subplot.hist(importances, alpha=0.8, label="Plane")

        expected_cut = importances.quantile(1 - self.percentile)
        subplot.axvline(expected_cut, color="black")

        if self.include_nexus:
            nexus_importances = data[-1]
            assert nexus_importances is not None, "Nexus edge weighs not calculated!"
            subplot.hist(nexus_importances, alpha=0.6, label="Nexus")
            subplot.legend()

    def _single_plot(
        self, style, subplot, distances, importances, nexus_importances=None
    ):
        data = (distances, importances, nexus_importances)
        if len(importances) != 0 and len(distances) != 0:
            {"scatter": self._scatter_plot, "histogram": self._histogram}[style](
                data, subplot
            )


class InteractiveEdgeVisuals:
    def __init__(
        self,
        plane,
        semantic_classes=["MIP", "HIP", "shower", "michel", "diffuse"],
        features=["1", "2", "3", "4", "5", "6"],
        feature_importance=False,
    ) -> None:
        self.plane = plane
        self.semantic_classes = semantic_classes
        self.features = features
        self.node_label_field = "node_mask" if feature_importance else "x"

    def _interative_edge(self, subgraph):
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = subgraph.nodes[edge[0]]["pos"]
            x1, y1 = subgraph.nodes[edge[1]]["pos"]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_traces = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            name="Edges",
            hoverinfo="none",
            line=dict(
                color="cornflowerblue",
            ),
            line_width=2,
        )

        return edge_traces

    def _interactive_nodes(self, subgraph, label: list[dict], class_label="0"):
        node_x = []
        node_y = []

        for node in subgraph.nodes():
            x, y = subgraph.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        trace_text = [
            f"{self.node_label_field}<br>"
            + "<br>".join(
                [
                    f"{feature}:{round(point_label[feature], 5)}"
                    for feature in point_label
                ]
            )
            for point_label in label
        ]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            name=class_label,
            text=trace_text,
            marker=dict(showscale=True, color="black"),
        )

        return node_trace

    def _grouped_interative_nodes(self, subgraph, label, node_label):
        labels = pd.DataFrame(label.values(), index=label.keys())
        labels.columns = ["label"]
        traces = []
        for unique in labels["label"].unique():
            nodes_to_include = labels[labels["label"] == unique].index.tolist()

            label_group = subgraph.__class__()
            label_group.add_nodes_from((n, subgraph.nodes[n]) for n in nodes_to_include)

            node_label = [node_label[node] for node in nodes_to_include]

            traces.append(
                self._interactive_nodes(
                    label_group, label=node_label, class_label=unique
                )
            )

        return traces

    def plot(self, graph, outdir=".", file_name="prediction_plot"):
        """
        Produce an interactive plot where the nodes are click-able and the field can be moved.
        Can only plot one field at a time, saves each result to an html.
        Args:
            plane (str, optional): Field to plot. Defaults to u.

        """

        subgraph, true_labels = make_subgraph_kx(
            graph, plane=self.plane, semantic_classes=self.semantic_classes
        )

        node_label = extract_node_weights(
            graph, plane=self.plane, node_field=self.node_label_field, scale=False
        )
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

        fig = go.Figure(data=data, layout=go.Layout(showlegend=True))
        fig.write_html(f"{outdir.rstrip('/')}/{file_name}.html")
