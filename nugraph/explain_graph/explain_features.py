import copy
import os
import json
from torch_geometric.explain import ModelConfig
import torch
from nugraph.explain_graph.algorithms.hetero_gnnexplainer import (
    HeteroExplainer,
    HeteroGNNExplainer,
)

from nugraph.explain_graph.explain_edges import GlobalGNNExplain
from nugraph.explain_graph.utils.node_visuals import NodeVisuals
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals, EdgeLengthDistribution
from nugraph.explain_graph.utils import get_masked_graph, MaskStrats


class GNNExplainerHits(GlobalGNNExplain):
    def __init__(
        self,
        data_path: str,
        out_path: str = "explainations/",
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

        index_list_file = f"{data_path.rstrip('.h5')}.json"
        assert os.path.exists(index_list_file)
        self.node_list = json.load(open(index_list_file))

        model_config = ModelConfig(
            mode="multiclass_classification", task_level="node", return_type="raw"
        )

        self.explainer = HeteroExplainer(
            model=self.model,
            algorithm=HeteroGNNExplainer(
                epochs=self.n_epochs,
                single_plane=False,
                plane=self.planes,
                nexus=True,
                nan_mask=True,
            ),
            explanation_type="model",
            model_config=model_config,
            node_mask_type="attributes",
            edge_mask_type="object",
        )
        self.explainations = {}
        self.classes = ["correct", "incorrect"]
        self.feature_names = feature_names
        self.criterion = (
            "hip_criteron" if "hip" in self.load.data_path else "michel_criteon"
        )

    def explain(self, data):
        try:
            len(data)
        except Exception:
            data = [data]

        indices = self.node_list.keys()
        if self.load.test:
            indices = [list(indices)[0]]

        graph_indices = range(len(indices))
        for graph_index, nodes, index in zip(
            graph_indices, self.node_list.values(), indices
        ):
            single_graph = {}
            self.metrics[index] = {}
            correct = nodes[self.criterion]["correct_hits"]
            incorrect = nodes[self.criterion]["incorrect_hits"]

            for name, active_nodes in zip(
                ["correct", "incorrect"], [correct, incorrect]
            ):
                if sum([len(active_nodes[plane]) for plane in self.planes]) != 0:
                    explain_graph = copy.deepcopy(data[graph_index])
                    explaination = self.explainer(
                        graph=explain_graph, nodes=active_nodes
                    )
                    metrics = self.calculate_metrics(explaination)

                    single_graph[name] = explaination
                    self.metrics[index][name] = metrics

            self.explainations[index] = single_graph
        return self.explainations

    def get_explanation_subgraph(self, explaination):
        edge_mask = explaination.collect("edge_mask")
        node_mask = explaination.collect("node_mask")
        return get_masked_graph(
            explaination,
            edge_mask=edge_mask,
            node_mask=node_mask,
            planes=self.planes,
            mask_strategy=MaskStrats.top_quartile,
            make_nodes_nan=True,
        )

    def visualize(self, explaination, *args, **kwargs):
        node_plotter = NodeVisuals(
            self.out_path,
            planes=self.planes,
            semantic_classes=self.classes,
            feature_names=self.feature_names,
        )
        edge_plotter = EdgeVisuals(planes=self.planes, semantic_classes=self.classes)
        edge_length_plotter = EdgeLengthDistribution(
            out_path=self.out_path,
            include_nexus=False,
            planes=self.planes,
            semantic_classes=self.classes,
        )

        for index, explain in explaination.items():
            ghost_plot = None
            file_name = f"{index}"
            if "correct" in explain.keys():
                ghost_plot = explain["correct"]

            elif "incorrect" in explain.keys():
                ghost_plot = explain["incorrect"]

            subgraphs = [subgraph_mask for subgraph_mask in explain.values()]
            subgraph_masked = [self.get_explanation_subgraph(g) for g in subgraphs]

            if len(subgraphs) != 0:
                edge_plotter.plot(
                    graph=subgraph_masked,
                    ghost_plot=ghost_plot,
                    outdir=self.out_path,
                    file_name=f"filter_subgraphs_{file_name}.png",
                    title=f"{self.criterion} Score: {round(self.node_list[index][self.criterion]['num_correct'], 4)}",
                    nexus_distribution=True,
                    class_plot=True,
                )

                node_plotter.plot(
                    style="hist",
                    graph=subgraphs,
                    split="plane",
                    file_name=f"{file_name}_histogram.png",
                )

                node_plotter.plot(
                    style="hist2d",
                    graph=subgraphs,
                    split="plane",
                    file_name=f"{file_name}_histd2_plane.png",
                )
                edge_length_plotter.plot(
                    graph=subgraphs,
                    style="scatter",
                    split="all",
                    file_name=f"{file_name}_length_corr.png",
                )

                edge_length_plotter.plot(
                    graph=subgraphs,
                    style="histogram",
                    split="all",
                    file_name=f"{file_name}_length_dist.png",
                )


class FilteredExplainedHits(GNNExplainerHits):
    def __init__(
        self,
        data_path: str,
        out_path: str = "explainations/",
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

    def visualize(self, explaination, file_name=None, *args, **kwargs):
        node_plotter = NodeVisuals(
            self.out_path,
            planes=self.planes,
            semantic_classes=self.classes,
            feature_names=self.feature_names,
        )

        subgraphs = [subgraph_mask for subgraph_mask in explaination.values()]
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
            explaination = self.explainer(graph=filtered_graph)

            self.single_visual(explaination, graph_index)
            metrics = self.calculate_metrics(explaination)
            self.metrics[str(len(self.metrics))] = metrics

            self.explainations.append(explaination)

        return self.explainations
