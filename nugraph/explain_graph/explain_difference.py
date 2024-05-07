import copy
import os
import json
from torch_geometric.explain import ModelConfig

from nugraph.explain_graph.algorithms.hetero_gnnexplaner import (
    HeteroExplainer,
    HeteroGNNExplainer,
)
from nugraph.explain_graph.algorithms.correct_gnn_explainer import CorrectGNNExplainer


from nugraph.explain_graph.explain_edges import GlobalGNNExplain
from nugraph.explain_graph.utils.node_visuals import NodeVisuals
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals, EdgeLengthDistribution
from nugraph.explain_graph.utils import get_masked_graph, MaskStrats


class GNNExplainerDifference(GlobalGNNExplain):
    def __init__(
        self,
        data_path: str,
        out_path: str = "./results/",
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

        if self.load.test:
            self.n_epochs = 2
        index_list_file = f"{data_path.rstrip('.h5')}.json"
        assert os.path.exists(index_list_file)

        self.node_list = json.load(open(index_list_file))
        self.criterion_name = "hip_criteron" if "hip" in data_path else "michel_criteon"
        self.explanations = {}

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
            self.init_explainer_algorithms()
            nodes = nodes[self.criterion_name]["incorrect_hits"]

            single_graph = {}
            self.metrics[index] = {}
            for explanation, name in zip(
                [self.explain_faithful, self.explain_correct], ["faithful", "correct"]
            ):
                explain_graph = copy.deepcopy(data[graph_index])
                e = explanation(graph=explain_graph, nodes=nodes)
                single_graph[name] = e
                self.metrics[index][name] = self.calculate_metrics(e)
                self.plot_loss(
                    file_name=f"{self.out_path.rstrip('/')}/explainer_loss_{index}.png"
                )

            self.explanations[index] = single_graph
        return self.explanations

    def init_explainer_algorithms(self):
        model_config = ModelConfig(
            mode="multiclass_classification", task_level="node", return_type="raw"
        )

        self.explain_faithful = HeteroExplainer(
            model=self.model,
            algorithm=HeteroGNNExplainer(
                epochs=self.n_epochs, single_plane=False, plane=self.planes, nexus=True
            ),
            explanation_type="model",
            model_config=model_config,
            node_mask_type="attributes",
            edge_mask_type="object",
        )
        self.explain_correct = HeteroExplainer(
            model=self.model,
            algorithm=CorrectGNNExplainer(
                epochs=self.n_epochs, single_plane=False, plane=self.planes, nexus=True
            ),
            explanation_type="model",
            model_config=model_config,
            node_mask_type="attributes",
            edge_mask_type="object",
        )

    def get_explanation_subgraph(self, explanation):
        edge_mask = explanation.collect("edge_mask")
        node_mask = explanation.collect("node_mask")

        return get_masked_graph(
            explanation,
            edge_mask=edge_mask,
            node_mask=node_mask,
            planes=self.planes,
            mask_strategy=lambda x, y: (MaskStrats.top_percentile(x, y, 0.15)),
        )

    def visualize(self, explanation, *args, **kwargs):
        visualization_classes = [
            "faithful",
            "correct",
        ]
        self.plot_loss(file_name=f"{self.out_path}/explainer_loss.png")
        node_plotter = NodeVisuals(
            self.out_path,
            planes=self.planes,
            semantic_classes=visualization_classes,
        )
        edge_plotter = EdgeVisuals(
            planes=self.planes, semantic_classes=visualization_classes
        )
        edge_length_plotter = EdgeLengthDistribution(
            out_path=self.out_path,
            include_nexus=False,
            planes=self.planes,
            semantic_classes=visualization_classes,
            percentile=0.15,
        )

        for index, explain in explanation.items():
            ghost_plot = explain["faithful"]
            subgraphs = [self.get_explanation_subgraph(g) for g in explain.values()]

            edge_plotter.plot(
                graph=subgraphs,
                ghost_plot=ghost_plot,
                outdir=self.out_path,
                file_name=f"filter_subgraphs_{index}.png",
                nexus_distribution=True,
                class_plot=True,
            )

            node_plotter.plot(
                style="hist",
                graph=subgraphs,
                split="all",
                file_name=f"feature_histogram_{index}.png",
            )

            node_plotter.plot(
                style="hist2d",
                graph=subgraphs,
                split="class",
                file_name=f"feature_hist2d_{index}.png",
            )

            edge_length_plotter.plot(
                graph=subgraphs,
                style="scatter",
                split="class",
                file_name=f"edge_length_corr_{index}.png",
            )

            edge_length_plotter.plot(
                graph=subgraphs,
                style="histogram",
                split="all",
                file_name=f"edge_length_dist_{index}.png",
            )

    def plot_loss(self, file_name):
        import matplotlib.pyplot as plt

        plt.close("all")
        fig, subplots = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

        subplots[0].plot(
            range(len(self.explain_faithful.algorithm.recall_loss_history)),
            self.explain_faithful.algorithm.recall_loss_history,
        )
        subplots[0].set_title("Faithful")

        subplots[1].plot(
            range(len(self.explain_correct.algorithm.loss_history)),
            self.explain_correct.algorithm.loss_history,
        )
        subplots[1].set_title("Correct")

        fig.supxlabel("Iteration")
        fig.supylabel("Recall Loss")

        plt.savefig(file_name)
        plt.close("all")
        plt.close("all")
