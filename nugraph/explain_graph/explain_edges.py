from collections import defaultdict
from typing import Iterable
from pynuml import io

import h5py
import torch
from torch_geometric.explain import ModelConfig
from torch_geometric.data import HeteroData
from datetime import datetime

from nugraph.explain_graph.explain import ExplainLocal
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals
from nugraph.explain_graph.algorithms.hetero_gnnexplainer import (
    HeteroExplainer,
    HeteroGNNExplainer,
)

from nugraph.explain_graph.algorithms.class_hetero_gnnexplainer import (
    MultiEdgeHeteroGNNExplainer,
)

from nugraph.explain_graph.utils.masking_utils import get_masked_graph, MaskStrats
from nugraph.explain_graph.utils.edge_visuals import (
    EdgeLengthDistribution,
)


class GlobalGNNExplain(ExplainLocal):
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
    ):
        self.planes = planes
        self.explainations = []
        self.masking_strat = MaskStrats.top_quartile

        super().__init__(
            data_path,
            out_path,
            checkpoint_path,
            batch_size,
            test,
            n_batches=n_batches,
            message_passing_steps=message_passing_steps,
            n_epochs=n_epochs,
        )

        model_config = ModelConfig(
            mode="multiclass_classification", task_level="node", return_type="raw"
        )

        self.explainer = HeteroExplainer(
            model=self.model,
            algorithm=HeteroGNNExplainer(epochs=self.n_epochs, plane=self.planes),
            explanation_type="model",
            model_config=model_config,
            node_mask_type=None,
            edge_mask_type="object",
        )

    def get_explanation_subgraph(self, explaination):
        edge_mask = explaination.collect("edge_mask")
        return get_masked_graph(
            explaination,
            edge_mask,
            planes=self.planes,
            mask_strategy=self.masking_strat,
        )

    def visualize(self, explaination, file_name=None, *args, **kwargs):
        for graph in explaination:
            if file_name is None:
                file_name = f"subgraph_{datetime.now().timestamp()}"

            subgraph = self.get_explanation_subgraph(graph)
            EdgeVisuals(planes=self.planes).plot(
                graph=subgraph,
                outdir=self.out_path,
                file_name=f"{file_name}.png",
                nexus_distribution=True,
            )

    def explain(self, data):
        try:
            len(data)
        except Exception:
            data = [data]

        if self.load.test:
            data = [data[0]]

        for graph in data:
            explaination = self.explainer(graph=graph)
            metrics = self.calculate_metrics(explaination)
            self.metrics[str(len(self.metrics))] = metrics

            self.explainations.append(explaination)

        return self.explainations

    def save(self):
        super().save()
        save_heterodata = HeteroData()
        if isinstance(self.explainations, list):
            for index in range(len(self.explainations)):
                save_heterodata[index] = self.explainations[index]
        interface = io.H5Interface(
            h5py.File(f"{self.out_path}/explaination_graphs.h5", "w")
        )
        interface.save_heterodata(save_heterodata)

    def plot_loss(self, file_name):
        import matplotlib.pyplot as plt

        plt.close("all")

        plt.plot(
            range(len(self.explainer.algorithm.loss_history)),
            self.explainer.algorithm.loss_history,
        )
        plt.xlabel("Iteration")
        plt.ylabel("Entropy Loss")

        plt.savefig(file_name)
        plt.close("all")


class ClasswiseGNNExplain(GlobalGNNExplain):
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
    ):
        super().__init__(
            data_path,
            out_path,
            checkpoint_path,
            batch_size,
            test,
            planes,
            n_batches=n_batches,
            message_passing_steps=message_passing_steps,
            n_epochs=n_epochs,
        )
        self.semantic_classes = ["MIP", "HIP", "shower", "michel", "diffuse"]
        model_config = ModelConfig(
            mode="multiclass_classification", task_level="node", return_type="raw"
        )

        self.explainer = HeteroExplainer(
            model=self.model,
            algorithm=MultiEdgeHeteroGNNExplainer(
                epochs=self.n_epochs, plane=self.planes
            ),
            explanation_type="model",
            model_config=model_config,
            node_mask_type="object",
            edge_mask_type="object",
        )

    def visualize(self, explaination):
        if not isinstance(explaination, Iterable):
            explaination = [explaination]

        for index, explain in enumerate(explaination):
            # Combine them all into one graph for the visuals
            subgraphs = []
            unmasked_subgraphs = []
            for sub_explain in explain.values():
                subgraph = self.get_explanation_subgraph(explaination=sub_explain)
                subgraphs.append(subgraph)
                unmasked_subgraphs.append(sub_explain)

            graph = explain[list(explain.keys())[0]]

            EdgeVisuals(planes=self.planes).plot(
                graph=subgraphs,
                ghost_plot=graph,
                outdir=self.out_path,
                file_name=f"filter_subgraphs_{index}.png",
                nexus_distribution=False,
                class_plot=True,
            )

            edge_plots = EdgeLengthDistribution(
                out_path=self.out_path,
                planes=self.planes,
                semantic_classes=self.semantic_classes,
                include_nexus=True,
            )

            edge_plots.plot(
                unmasked_subgraphs,
                style="histogram",
                split="all",
                file_name="class_edge_distribution.png",
            )
            edge_plots.plot(
                unmasked_subgraphs,
                style="scatter",
                split="all",
                file_name="length_correlation.png",
            )

    def calculate_metrics(self, explainations):
        metrics = {}
        for key in explainations:
            metrics[key] = super().calculate_metrics(explainations[key])
        return metrics


class FilteredExplainEdges(GlobalGNNExplain):
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

        self.classes = ("MIP", "HIP", "shower", "michel", "diffuse")
        self.explanations = defaultdict(dict)
        self.metrics = defaultdict(dict)

    def filter_graph(self, graph):
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

        return {"correct": correct, "incorrect": incorrect}

    def visualize(self, explaination, file_name=None, *args, **kwargs):
        edge_length_plotter = EdgeLengthDistribution(
            out_path=self.out_path,
            include_nexus=False,
            planes=self.planes,
            semantic_classes=("correct", "incorrect"),
        )
        edge_plotter = EdgeVisuals(
            planes=self.planes, semantic_classes=("correct", "incorrect")
        )

        for index, graph in explaination.items():
            subgraph_masked = {
                key: self.get_explanation_subgraph(g) for key, g in graph.items()
            }
            file_name = (
                f"{file_name}_{index}"
                if file_name is not None
                else f"filtered_event_{index}"
            )

            for name, subgraph in subgraph_masked.items():
                edge_plotter.plot(
                    subgraph,
                    outdir=self.out_path,
                    class_plot=False,
                    file_name=f"{file_name}_{name}.png",
                )

            edge_length_plotter.plot(
                graph=list(subgraph_masked.values()),
                style="scatter",
                split="all",
                file_name=f"{file_name}_length_corr.png",
            )

            edge_length_plotter.plot(
                graph=list(subgraph_masked.values()),
                style="histogram",
                split="all",
                file_name=f"{file_name}_length_dist.png",
            )

    def explain(self, data):
        try:
            len(data)
        except Exception:
            data = [data]

        if True:
            data = [data[0]]

        for graph_index, graph in enumerate(data):
            filtered = self.filter_graph(graph)
            for concept, nodes in filtered.items():
                explaination = self.explainer(graph=graph, nodes=nodes)

                self.metrics[graph_index][concept] = self.calculate_metrics(
                    explaination
                )
                self.explanations[graph_index][concept] = explaination

        return self.explanations
