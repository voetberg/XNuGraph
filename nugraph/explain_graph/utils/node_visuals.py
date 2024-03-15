import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals


class NodeVisuals:
    def __init__(
        self,
        out_path=".",
        planes=["u", "v", "y"],
        semantic_classes=["MIP", "HIP", "shower", "michel", "diffuse"],
        percentile=0.3,
        ghost_overall=False,
        n_bins=30,
    ) -> None:
        self.out_path = out_path
        self.planes = planes
        self.semantic_classes = semantic_classes
        self.percentile = percentile
        self.draw_ghost = ghost_overall
        self.n_bins = n_bins

    def plot(self, style, graph, split="class", file_name="plot.png"):
        assert style in ["hist", "hist2d", "heat"]

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
        }[split](graph, style, subplots)

        figure.suptitle(split)
        if style in ["heat", "hist2d"]:
            figure.supxlabel("Time")
            figure.supylabel("Wire Number")

        plt.tight_layout()
        plt.savefig(f"{self.out_path.rstrip('/')}/{file_name}")

    def _single_plot(self, style, subplot, importances, position):
        {
            "hist": self._hist,
            "hist2d": self._hist_2d,
            "heat": self._heat,
        }[style](subplot, importances, position)

    def _hist(self, subplot, importances, position):
        for row in range(importances.shape[1]):
            subplot.hist(importances[:, row], alpha=0.65, label=row)
        subplot.legend()

    def _hist_2d(self, subplot, importances, position):
        x, y = position[:, 0], position[:, 1]
        subplot.hist2d(x, y, bins=self.n_bins, weights=importances)

    def _heat(self, subplot, importances, position):
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
        subplot.contourf(x, y, z, levels=contours)

    def _plot_classwise(self, graph, style, subplots):
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
                )
                subplots[0][label_index].set_title(label)

    def _plot_planewise(self, graph, style, subplots):
        if style != "heat":
            plane_importances = [
                graph[index].collect("node_mask") for index in graph.keys()
            ]
            plane_positions = [graph[index].collect("pos") for index in graph.keys()]

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
                    )
                    subplots[0][plane_index].set_title([plane])
                    if self.draw_ghost and style != "hist":
                        EdgeVisuals(planes=self.planes).draw_ghost_plot(
                            graph[0], plane=plane, axes=subplots[feature][plane_index]
                        )
        else:
            print("heatmap plots not supported for planewise splits")

    def _plot_all_together(self, graph, style, subplots):
        importances = graph.collect("node_mask")
        positions = graph.collect("position")
        importance = np.concatenate([importances[plane] for plane in self.planes])
        position = np.concatenate([positions[plane] for plane in self.planes])

        self._single_plot(style, subplots, importance, position)

    def _plot_all_seperate(self, graph, style, subplots):
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
                if self.draw_ghost and style != "hist":
                    EdgeVisuals(planes=self.planes).draw_ghost_plot(
                        subgraph, plane=plane, axes=subplot_row[plane_index]
                    )
