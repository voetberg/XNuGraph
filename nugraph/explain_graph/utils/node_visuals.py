import matplotlib.pyplot as plt


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

        n_rows = 1 if not isinstance(graph, dict) else len(graph.keys())
        n_cols = len(self.planes)

        figure, subplots = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows)
        )

        self._plot_classwise(graph, style, subplots, key_hits)

        figure.suptitle(split)
        if style in ["heat", "hist2d"]:
            figure.supxlabel("Time")
            figure.supylabel("Wire Number")

        plt.tight_layout()
        plt.savefig(f"{self.out_path.rstrip('/')}/{file_name}")
        plt.close()

    def _single_plot(self, style, subplot, importances, position, y_label=""):
        {
            "hist": self._hist,
            "hist2d": self._hist_2d,
        }[style](subplot, importances, position, y_label)

    def _hist(self, subplot, importances, position, y_label):
        try:
            if importances.shape[0] > importances.shape[1]:
                importances = importances.T
        except IndexError:
            pass
        subplot.hist(importances, alpha=0.65)
        subplot.set_ylabel(y_label)

        subplot.legend()

    def _hist_2d(self, subplot, importances, position, y_label):
        x, y = position[:, 0], position[:, 1]
        subplot.set_ylabel(y_label)
        subplot.hist2d(x, y, bins=self.n_bins, weights=importances.ravel())

    def _plot_classwise(self, graph, style, subplots, key_hits=None):
        if not isinstance(graph, dict):
            graph = {"": graph}

        for label_index, (label, subgraph) in enumerate(graph.items()):
            importances = subgraph.collect("node_mask")
            positions = subgraph.collect("pos")

            for plane_index, plane in enumerate(self.planes):
                self._single_plot(
                    style,
                    subplots[label_index][plane_index],
                    importances[plane],
                    positions[plane],
                    y_label=label,
                )
                subplots[0][plane_index].set_title(plane)
