import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
from nugraph.explain_graph.utils.visuals_common import highlight_nodes


class NodeVisuals:
    def __init__(
        self,
        out_path=".",
        planes=["u", "v", "y"],
        semantic_classes=["MIP", "HIP", "shower", "michel", "diffuse"],
        feature_names=["Wire", "Peak", "Integral", "RMS"],
        percentile=0.3,
        n_bins=30,
    ) -> None:
        self.out_path = out_path
        self.planes = planes
        self.semantic_classes = semantic_classes
        self.percentile = percentile
        self.n_bins = n_bins
        self.feature_names = feature_names

    def plot(self, style, graph, split="class", file_name="plot.png", key_hits=None):
        assert style in ["hist", "hist2d", "heat", "event"]

        n_features = graph[0].collect("x")[self.planes[0]].shape[1]
        n_rows = {
            "class": n_features,
            "plane": n_features,
            "all": len(self.planes),
            "none": 1,
        }[split]
        n_cols = {
            "class": len(self.semantic_classes),
            "plane": len(self.planes),
            "all": len(self.semantic_classes),
            "none": 1,
        }[split]

        figure, subplots = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows)
        )

        {
            "class": self._plot_classwise,
            "plane": self._plot_planewise,
            "all": self._plot_all_seperate,
            "none": self._plot_all_together,
        }[split](graph, style, subplots, key_hits)

        figure.suptitle(split)
        if style in ["heat", "hist2d"]:
            figure.supxlabel("Time")
            figure.supylabel("Wire Number")

        plt.tight_layout()
        plt.savefig(f"{self.out_path.rstrip('/')}/{file_name}")

    def _single_plot(self, style, subplot, importances, position, y_label=""):
        {
            "hist": self._hist,
            "hist2d": self._hist_2d,
            "heat": self._heat,
            "event": self._event,
        }[style](subplot, importances, position, y_label)

    def _hist(self, subplot, importances, position, y_label):
        if importances.shape[0] > importances.shape[1]:
            importances = importances.T
        subplot.hist(importances, alpha=0.65)
        subplot.set_ylabel(y_label)

        subplot.legend()

    def _hist_2d(self, subplot, importances, position, y_label):
        x, y = position[:, 0], position[:, 1]
        subplot.set_ylabel(y_label)
        subplot.hist2d(x, y, bins=self.n_bins, weights=importances)

    def _heat(self, subplot, importances, position, y_label):
        x = np.linspace(
            position[:, 0].min(), position[:, 0].max(), num=len(importances.ravel())
        )
        y = np.linspace(
            position[:, 1].min(), position[:, 1].max(), num=len(importances.ravel())
        )
        X, Y = np.meshgrid(x, y)

        interoplatator = Rbf(x, y, importances.ravel())
        z = interoplatator(X, Y)

        contours = np.linspace(z.min(), z.max(), self.n_bins)
        subplot.set_ylabel(y_label)
        subplot.contourf(x, y, z, levels=contours)

    def _event(self, subplot, importances, position):
        x, y = position[:, 0], position[:, 1]
        subplot.scatter(x, y, c=importances)

    def _plot_classwise(self, graph, style, subplots, key_hits=None):
        for label_index, label in enumerate(self.semantic_classes):
            subgraph = graph[label_index]
            i = subgraph.collect("node_mask")
            positions = subgraph.collect("pos")
            importances = np.concatenate([i[plane] for plane in self.planes])
            position = np.concatenate([positions[plane] for plane in self.planes])

            for feature in range(importances.shape[1]):
                self._single_plot(
                    style,
                    subplots[feature][label_index],
                    importances[:, feature],
                    position,
                    self.feature_names[feature],
                )
            subplots[0][label_index].set_title(label)

    def _plot_planewise(self, graph, style, subplots, key_hits=None):
        if style != "heat":
            try:
                plane_importances = [
                    graph[index].collect("node_mask") for index in graph.keys()
                ]
                plane_positions = [
                    graph[index].collect("pos") for index in graph.keys()
                ]

            except AttributeError:
                plane_importances = [g.collect("node_mask") for g in graph]
                plane_positions = [g.collect("pos") for g in graph]

            for plane_index, plane in enumerate(self.planes):
                importance = np.concatenate(
                    [graph[plane] for graph in plane_importances]
                )
                position = np.concatenate([graph[plane] for graph in plane_positions])

                for feature in range(importance.shape[1]):
                    self._single_plot(
                        style,
                        subplots[feature][plane_index],
                        importance[:, feature],
                        position,
                        self.feature_names[feature],
                    )
                    subplots[0][plane_index].set_title([plane])

                    if (
                        (key_hits is not None)
                        and (style != "hist")
                        and (len(key_hits.keys()) == 2)
                    ):
                        plane_hits = (
                            key_hits[list(key_hits.keys())[0]][plane]
                            if sum(
                                [
                                    len(key_hits[list(key_hits.keys())[0]][plane])
                                    for plane in self.planes
                                ]
                            )
                            != 0
                            else key_hits[list(key_hits.keys())[1]][plane]
                        )
                        highlight_nodes(
                            graph=graph[0],
                            node_list=plane_hits,
                            plane=plane,
                            axes=subplots[feature][plane_index],
                        )

        else:
            print("heatmap plots not supported for planewise splits")

    def _plot_all_together(self, graph, style, subplots, key_hits=None):
        importances = graph.collect("node_mask")
        positions = graph.collect("position")
        importance = np.concatenate([importances[plane] for plane in self.planes])
        position = np.concatenate([positions[plane] for plane in self.planes])

        self._single_plot(style, subplots, importance, position)

    def _plot_all_seperate(self, graph, style, subplots, key_hits=None):
        for label_index, label in enumerate(self.semantic_classes):
            subgraph = graph[label_index]
            subplot_row = subplots[:, label_index]
            subplot_row[0].set_title(label)

            importances = subgraph.collect("node_mask")
            positions = subgraph.collect("pos")
            for plane_index, plane in enumerate(self.planes):
                self._single_plot(
                    style,
                    subplot_row[plane_index],
                    importances[plane],
                    positions[plane],
                )
                subplot_row[plane_index].set_ylabel(plane)

                if (key_hits is not None) and (style == "hist2d"):
                    highlight_nodes(
                        graph=graph[0][plane],
                        node_list=key_hits[label][plane],
                        plane=plane,
                        axes=subplot_row[plane_index],
                    )
