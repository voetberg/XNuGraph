from torch_geometric.explain import ModelConfig

from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroExplainer
from nugraph.explain_graph.algorithms.class_hetero_gnnexplainer import (
    MultiEdgeHeteroGNNExplainer,
)
from nugraph.explain_graph.gnn_explain_edges import GlobalGNNExplain

from nugraph.explain_graph.utils.node_visuals import NodeVisuals


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

    def get_explaination_subgraph(self, explaination):
        return explaination

    def visualize(self, explaination, file_name="explaination_graph"):
        plotter = NodeVisuals(self.out_path, planes=self.planes)
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