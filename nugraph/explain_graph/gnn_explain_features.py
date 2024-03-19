import copy
import os
import json
from torch_geometric.explain import ModelConfig

from nugraph.explain_graph.algorithms.hetero_gnnexplaner import (
    HeteroExplainer,
    HeteroGNNExplainer,
)
from nugraph.explain_graph.algorithms.class_hetero_gnnexplainer import (
    MultiEdgeHeteroGNNExplainer,
)
from nugraph.explain_graph.gnn_explain_edges import GlobalGNNExplain
from nugraph.explain_graph.utils.node_visuals import NodeVisuals
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals, EdgeLengthDistribution
from nugraph.explain_graph.utils import get_masked_graph, MaskStrats


class GNNExplainFeatures(GlobalGNNExplain):
    def __init__(
        self,
        data_path: str,
        out_path: str = "explainations/",
        checkpoint_path: str = None,
        batch_size: int = 16,
        test: bool = False,
        planes=["u", "v", "y"],
        message_passing_steps=5,
        n_epochs=200,
    ):
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
            algorithm=MultiEdgeHeteroGNNExplainer(
                epochs=self.n_epochs, single_plane=False, plane=self.planes, nexus=False
            ),
            explanation_type="model",
            model_config=model_config,
            node_mask_type="attributes",
        )
        self.classes = ["correct", "incorrect"]

    def get_explaination_subgraph(self, explaination):
        return explaination

    def visualize(self, explaination, file_name="explaination_graph"):
        plotter = NodeVisuals(self.out_path, planes=self.planes, ghost_overall=True)
        for graph in explaination:
            plotter.plot(
                style="hist",
                graph=graph,
                split="all",
                file_name=f"{file_name}_histogram.png",
            )

            plotter.plot(
                style="heat",
                graph=graph,
                split="all",
                file_name=f"{file_name}_heat_all.png",
            )

            plotter.plot(
                style="heat",
                graph=graph,
                split="class",
                file_name=f"{file_name}_heat_class.png",
            )

            plotter.plot(
                style="hist2d",
                graph=graph,
                split="class",
                file_name=f"{file_name}_hist2d_class.png",
            )
            plotter.plot(
                style="hist2d",
                graph=graph,
                split="plane",
                file_name=f"{file_name}_histd2_plane.png",
            )

    def calculate_metrics(self, explainations):
        all_metrics = {}
        for index, class_explain in explainations.items():
            all_metrics[index] = super().calculate_metrics(class_explain)
        return all_metrics


class GNNExplainerHits(GlobalGNNExplain):
    def __init__(
        self,
        data_path: str,
        out_path: str = "explainations/",
        checkpoint_path: str = None,
        batch_size: int = 16,
        test: bool = False,
        planes=["u", "v", "y"],
        message_passing_steps=5,
        n_epochs=250,
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
                epochs=self.n_epochs, single_plane=False, plane=self.planes, nexus=True
            ),
            explanation_type="model",
            model_config=model_config,
            node_mask_type="attributes",
            edge_mask_type="object",
        )
        self.explainations = {}
        self.classes = ["correct", "incorrect"]

    def explain(self, data):
        try:
            len(data)
        except Exception:
            data = [data]

        indices = self.node_list.keys()
        graph_indices = range(len(indices))
        for graph_index, nodes, index in zip(
            graph_indices, self.node_list.values(), indices
        ):
            single_graph = {}
            self.metrics[index] = {}

            for criteron, n in nodes.items():
                single_graph[criteron] = {}
                self.metrics[index][criteron] = {}

                correct = n["correct_hits"]
                incorrect = n["incorrect_hits"]
                for name, active_nodes in zip(
                    ["correct", "incorrect"], [correct, incorrect]
                ):
                    if sum([len(active_nodes[plane]) for plane in self.planes]) != 0:
                        explain_graph = copy.deepcopy(data[graph_index])
                        explaination = self.explainer(
                            graph=explain_graph, nodes=active_nodes
                        )
                        metrics = self.calculate_metrics(explaination)

                        single_graph[criteron][name] = explaination
                        self.metrics[index][criteron][name] = metrics

            self.explainations[index] = single_graph
        return self.explainations

    def get_explaination_subgraph(self, explaination):
        edge_mask = explaination.collect("edge_mask")
        node_mask = explaination.collect("node_mask")
        return get_masked_graph(
            explaination,
            edge_mask=edge_mask,
            node_mask=node_mask,
            planes=self.planes,
            mask_strategy=MaskStrats.top_quartile,
        )

    def visualize(self, explaination, *args, **kwargs):
        node_plotter = NodeVisuals(
            self.out_path,
            planes=self.planes,
            ghost_overall=True,
            semantic_classes=self.classes,
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
            for key, value in explain.items():
                file_name = f"{key}_{index}"
                if "correct" in explain[key].keys():
                    ghost_plot = explain[key]["correct"]

                elif "incorrect" in explain[key].keys():
                    ghost_plot = explain[key]["incorrect"]

                subgraphs = [subgraph_mask for subgraph_mask in value.values()]
                subgraph_masked = [self.get_explaination_subgraph(g) for g in subgraphs]

                if len(subgraphs) != 0:
                    edge_plotter.plot(
                        graph=subgraph_masked,
                        ghost_plot=ghost_plot,
                        outdir=self.out_path,
                        file_name=f"filter_subgraphs_{file_name}.png",
                        title=f"{key} Score: {round(self.node_list[index][key]['num_correct'], 4)}",
                        nexus_distribution=False,
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
