from typing import Iterable
from pynuml import io

import h5py
from torch_geometric.explain import ModelConfig
from torch_geometric.data import HeteroData
from datetime import datetime

from nugraph.explain_graph.explain import ExplainLocal
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals
from nugraph.explain_graph.algorithms.hetero_gnnexplaner import (
    HeteroGNNExplainer,
    HeteroExplainer,
)
from nugraph.explain_graph.algorithms.class_hetero_gnnexplainer import (
    MultiEdgeHeteroGNNExplainer,
)
from nugraph.explain_graph.algorithms.prune_gnn_explainer import (
    NonTrainedHeteroGNNExplainer,
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
            node_mask_type="object",
            edge_mask_type="object",
        )

    def get_explaination_subgraph(self, explaination):
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

            subgraph = self.get_explaination_subgraph(graph)
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
                subgraph = self.get_explaination_subgraph(explaination=sub_explain)
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


class GNNExplainerPrune(GlobalGNNExplain):
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
            message_passing_steps=message_passing_steps,
            n_epochs=n_epochs,
        )
        model_config = ModelConfig(
            mode="multiclass_classification", task_level="node", return_type="raw"
        )

        self.explainer = HeteroExplainer(
            model=self.model,
            algorithm=NonTrainedHeteroGNNExplainer(
                epochs=self.n_epochs, plane=self.planes
            ),
            explanation_type="model",
            model_config=model_config,
            node_mask_type="object",
        )

    def explain(self, data):
        explaination = self.explainer(graph=data)
        return explaination

    def get_explaination_subgraph(self, explaination):
        edge_mask = explaination["edge_mask"]
        masked_graph = get_masked_graph(
            explaination["graph"], edge_mask=edge_mask, planes=self.planes
        )
        masked_graph["edge_mask"] = edge_mask
        return masked_graph
