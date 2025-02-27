import copy
from torch_geometric.explain import ModelConfig
import torch
from nugraph.explain_graph.algorithms.hetero_gnnexplainer import (
    HeteroExplainer,
    HeteroGNNExplainer,
)

from nugraph.explain_graph.explain_edges import GlobalGNNExplain
from nugraph.explain_graph.utils.node_visuals import NodeVisuals
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals
from nugraph.explain_graph.utils import get_masked_graph, MaskStrats


class GNNExplainerHits(GlobalGNNExplain):
    def __init__(
        self,
        data_path: str,
        out_path: str = "explanations/",
        node_attribution: bool = False,
        checkpoint_path: str = None,
        batch_size: int = 16,
        test: bool = False,
        planes=["u", "v", "y"],
        feature_names=["Wire", "Peak", "Integral", "RMS"],
        message_passing_steps=5,
        n_epochs=150,
    ) -> None:
        self.planes = planes
        super().__init__(
            data_path,
            out_path,
            checkpoint_path,
            batch_size,
            test,
            message_passing_steps=message_passing_steps,
            n_epochs=n_epochs,
        )

        model_config = ModelConfig(
            mode="multiclass_classification", task_level="node", return_type="raw"
        )

        self.explainer = HeteroExplainer(
            model=self.model,
            algorithm=HeteroGNNExplainer(
                epochs=self.n_epochs,
                single_plane=False,
                plane=self.planes,
                nexus=False,
                nan_mask=True,
            ),
            explanation_type="model",
            model_config=model_config,
            node_mask_type="attributes" if node_attribution else "object",
            edge_mask_type=None,
        )
        self.explanations = {}
        self.classes = ["correct", "incorrect"]
        self.feature_names = feature_names
        self.node_attribution = node_attribution

    def explain(self, data):
        try:
            len(data)
        except Exception:
            data = [data]

        test = True
        if test:
            data = [data[0]]

        for index, graph in enumerate(data):
            single_graph = {}
            self.metrics[index] = {}

            predictions = self.model(*self.load.unpack(graph))

            predictions = {
                p: torch.argmax(predictions["x_semantic"][p], axis=-1).to(int)
                for p in self.planes
            }
            labels = graph.collect("y_semantic")

            correct = {
                plane: [
                    id
                    for id, label in enumerate(labels[plane])
                    if (predictions[plane][id] == label and label != -1)
                ]
                for plane in self.planes
            }
            incorrect = {
                plane: [
                    id
                    for id, label in enumerate(labels[plane])
                    if (predictions[plane][id] != label and label != -1)
                ]
                for plane in self.planes
            }

            for name, active_nodes in zip(
                ["correct", "incorrect"], [correct, incorrect]
            ):
                if sum([len(active_nodes[plane]) for plane in self.planes]) != 0:
                    explain_graph = copy.deepcopy(graph)
                    explanations = self.explainer(
                        graph=explain_graph, nodes=active_nodes
                    )
                    metrics = self.calculate_metrics(explanations)

                    single_graph[name] = explanations
                    self.metrics[index][name] = metrics

            self.explanations[index] = single_graph
        return self.explanations

    def get_explanation_subgraph(self, explanation):
        node_mask = explanation.collect("node_mask")
        return get_masked_graph(
            explanation,
            edge_mask=None,
            node_mask=node_mask,
            planes=self.planes,
            mask_strategy=MaskStrats.top_quartile,
            make_nodes_nan=True,
        )

    def visualize(self, explanation, *args, **kwargs):
        if self.node_attribution:
            features = self.feature_names
        else:
            features = self.classes
        node_plotter = NodeVisuals(
            self.out_path,
            planes=self.planes,
            semantic_classes=self.classes,
            feature_names=features,
        )
        edge_plotter = EdgeVisuals(planes=self.planes, semantic_classes=self.classes)

        for index, explain in explanation.items():
            ghost_plot = None
            file_name = f"{index}"
            if "correct" in explain.keys():
                ghost_plot = explain["correct"]

            elif "incorrect" in explain.keys():
                ghost_plot = explain["incorrect"]

            subgraphs = [subgraph_mask for subgraph_mask in explain.values()]
            subgraph_masked = {
                c: self.get_explanation_subgraph(g) for c, g in explain.items()
            }

            if len(subgraphs) != 0:
                if not self.node_attribution:
                    edge_plotter.plot(
                        graph=subgraph_masked,
                        ghost_plot=ghost_plot,
                        outdir=self.out_path,
                        file_name=f"filter_subgraphs_{file_name}.png",
                        title="",
                        nexus_distribution=False,
                        class_plot=True,
                    )
                    node_plotter.plot(
                        style="hist2d",
                        graph=explain,
                        split="plane",
                        file_name=f"{file_name}_histd2_plane.png",
                    )
                else:
                    for name, e in explain.items():
                        mask = e.collect("node_mask")
                        n_features = mask[self.planes[0]].shape[1]
                        items = {}
                        for feature in range(n_features):
                            for plane in self.planes:
                                e[plane]["node_mask"] = mask[plane][:, feature]
                            items[feature + 1] = e

                        node_plotter.plot(
                            style="hist2d",
                            graph=items,
                            split="plane",
                            file_name=f"{file_name}_{name}_histd2_plane.png",
                        )

                node_plotter.plot(
                    style="hist",
                    graph=explain,
                    split="plane",
                    file_name=f"{file_name}_histogram.png",
                )


class FilteredExplainedHits(GNNExplainerHits):
    def __init__(
        self,
        data_path: str,
        out_path: str = "explanations/",
        checkpoint_path: str = None,
        batch_size: int = 16,
        test: bool = False,
        planes=["u", "v", "y"],
        n_batches=None,
        message_passing_steps=5,
        n_epochs=200,
        node_filter=True,
        background_filter=True,
    ):
        super().__init__(
            data_path,
            out_path,
            checkpoint_path,
            batch_size,
            test,
            planes,
            n_batches,
            message_passing_steps,
            n_epochs,
        )

        self.background_filter = background_filter
        self.node_list = node_filter

    def filter_graph(self, graph):
        if self.background_filter:
            mask = {
                plane: torch.logical_not(
                    torch.isin(graph.collect("x_filter")[plane], torch.tensor([-1]))
                )
                for plane in ["u", "v", "y"]
            }

        else:
            mask = {
                plane: torch.logical_not(
                    torch.isin(
                        graph.collect("y_semantic")[plane], torch.tensor([-1, 2, 4])
                    )
                )
                for plane in ["u", "v", "y"]
            }

        return get_masked_graph(
            graph,
            node_mask=mask,
            planes=self.planes,
            mask_strategy=self.masking_strat,
        )

    def visualize(self, explanation, file_name=None, *args, **kwargs):
        node_plotter = NodeVisuals(
            self.out_path,
            planes=self.planes,
            semantic_classes=self.classes,
            feature_names=self.feature_names,
        )

        subgraphs = [subgraph_mask for subgraph_mask in explanation.values()]
        subgraph_masked = [self.get_explanation_subgraph(g) for g in subgraphs]

        if len(subgraphs) != 0:
            node_plotter.plot(
                style="hist",
                graph=subgraph_masked,
                split="plane",
                file_name=f"{file_name}_histogram.png",
            )
            node_plotter.plot(
                style="hist2d",
                graph=subgraph_masked,
                split="plane",
                file_name=f"{file_name}_histd2_plane.png",
            )

    def single_visual(self, explain, index):
        subgraph = self.get_explanation_subgraph(explain)
        EdgeVisuals(planes=self.planes).plot(
            graph=subgraph,
            outdir=self.out_path,
            file_name=f"subgraph_{index}.png",
            nexus_distribution=True,
        )

    def explain(self, data):
        try:
            len(data)
        except Exception:
            data = [data]

        if self.load.test:
            data = [data[0]]

        for graph_index, graph in enumerate(data):
            filtered_graph = self.filter_graph(graph=graph)
            explanation = self.explainer(graph=filtered_graph)

            self.single_visual(explanation, graph_index)
            metrics = self.calculate_metrics(explanation)
            self.metrics[str(len(self.metrics))] = metrics

            self.explanations.append(explanation)

        return self.explanations
