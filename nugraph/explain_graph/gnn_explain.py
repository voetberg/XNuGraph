from typing import Iterable
from nugraph.explain_graph.explain import ExplainLocal
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals
import json 

from torch_geometric.explain import ModelConfig
from datetime import datetime 
from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroGNNExplainer, HeteroExplainer
from nugraph.explain_graph.class_hetero_gnnexplainer import MultiEdgeHeteroGNNExplainer
from nugraph.explain_graph.algorithms.prune_gnn_explainer import NonTrainedHeteroGNNExplainer
from nugraph.explain_graph.utils.masking_utils import get_masked_graph, MaskStrats
from nugraph.explain_graph.utils.edge_visuals import (
    EdgeVisuals, 
    InteractiveEdgeVisuals, 
    make_subgraph_kx, 
    EdgeLengthDistribution
)

import matplotlib.pyplot as plt 
import h5py 

class GlobalGNNExplain(ExplainLocal): 
    def __init__(self, data_path: str, out_path: str = "explainations/", checkpoint_path: str = None, batch_size: int = 16, test: bool = False, planes=['u', 'v', 'y'], n_batches=None, message_passing_steps=5):
        self.planes = planes
        self.explainations = []
        self.masking_strat = MaskStrats.top_quartile

        super().__init__(data_path, out_path, checkpoint_path, batch_size, test, n_batches=n_batches, message_passing_steps=message_passing_steps)

        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node', 
            return_type="raw")
        
        self.explainer = HeteroExplainer(
            model=self.model, 
            algorithm=HeteroGNNExplainer(epochs=3, plane=self.planes), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="object",
            edge_mask_type="object",
        )

    def get_explaination_subgraph(self, explaination):
        edge_mask = explaination.collect("edge_mask")
        return get_masked_graph(
            explaination, edge_mask, planes=self.planes, mask_strategy=self.masking_strat
        )
    
    def visualize(self, explaination, file_name=None, *args, **kwargs):
        for graph in explaination: 
            if file_name is None: 
                file_name = f"subgraph_{datetime.now().timestamp()}"

            subgraph = self.get_explaination_subgraph(graph)
            EdgeVisuals(planes=self.planes).plot(graph=subgraph, outdir=self.out_path, file_name=f"{file_name}.png", nexus_distribution=True)

            json.dump(self.metrics, open(f"{self.out_path}/metrics_{file_name}.json", 'w'))

    def explain(self, data):
        try:
            len(data)
        except: 
            data = [data]

        for graph in data: 
            explaination = self.explainer(graph=graph)
            metrics = self.calculate_metrics(explaination)
            self.metrics[str(len(self.metrics))] = metrics

            self.explainations.append(explaination)

        return self.explainations
    
    def save(self, file_name: str = None):
        super().save(file_name)

        with h5py.File(f"{self.out_path}/{file_name}.h5", 'w') as f: 
            f.create_dataset(name="results", data=self.explainations)
            
        f.close()
    
class ClasswiseGNNExplain(GlobalGNNExplain): 
    def __init__(self, data_path: str, out_path: str = "explainations/", checkpoint_path: str = None, batch_size: int = 16, test: bool = False, planes=['u', 'v', 'y'], n_batches=None, message_passing_steps=5):
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test, planes, n_batches=n_batches, message_passing_steps=message_passing_steps)
        self.semantic_classes = ['MIP','HIP','shower','michel','diffuse']
        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node', 
            return_type="raw")
        
        self.explainer = HeteroExplainer(
            model=self.model, 
            algorithm=MultiEdgeHeteroGNNExplainer(epochs=2, plane=self.planes), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="object",
            edge_mask_type="object",
        )

    def visualize(self, explaination, file_name):
        if file_name is None: 
            file_name = f"subgraph_{datetime.now().timestamp().split('.')[-1]}"

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
                file_name=f"{file_name}_{index}.png", 
                nexus_distribution=False, 
                class_plot=True)

            edge_plots = EdgeLengthDistribution(
                out_path=self.out_path, 
                planes=self.planes, 
                semantic_classes=self.semantic_classes, 
                include_nexus=True)

            edge_plots.plot(
                unmasked_subgraphs, 
                style='histogram', 
                split='all', 
                file_name="class_edge_distribution.png")
            
            edge_plots.plot(
                unmasked_subgraphs, 
                style='scatter', 
                split='all', 
                file_name="length_correlation.png")


            self.explainer.algorithm.plot_loss(
                f"{self.out_path.rstrip('/')}/explainer_loss.png"
            )

    def calculate_metrics(self, explainations):
        metrics = {}
        for key in explainations: 
            metrics[key] = super().calculate_metrics(explainations[key])
        return metrics
    
class GNNExplainFeatures(GlobalGNNExplain): 
    def __init__(self, data_path: str, out_path: str = "explainations/", checkpoint_path: str = None, batch_size: int = 16, test: bool = False, planes=['u', 'v', 'y'],message_passing_steps=5):
        self.planes = planes
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test, message_passing_steps=message_passing_steps)

        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node', 
            return_type="raw")
        
        self.explainer = HeteroExplainer(
            model=self.model, 
            algorithm=HeteroGNNExplainer(epochs=100, single_plane=False, plane=self.planes), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="attributes", 
        )

    def _importance_plot(self, subgraph, file_name): 
        plot_engine = EdgeVisuals()
        _, subplots = plt.subplots(2, 3, figsize=(16*3, 16*2))
        subgraph = subgraph['graph']
        for index, plane in enumerate(self.planes): 
            subgraph_kx = make_subgraph_kx(subgraph, plane=plane)
            node_list = subgraph_kx.nodes 
            subplots[0, index].set_title(plane)

            plot_engine.plot_graph(subgraph, subgraph_kx, plane, node_list, subplots[0, index])

            importance = subgraph['node_mask'][plane].mean(axis=0)
            subplots[1, index].bar(x=range(len(importance)), height=importance)
            subgraph[plane]['node_mask'] = subgraph['node_mask'][plane]
            
            # Produce interactive plots at the same time. 
            InteractiveEdgeVisuals(
                plane=plane, 
                feature_importance=True
            ).plot(subgraph, outdir=self.out_path, file_name=f"interactive_{plane}")

        plt.savefig(f"{self.out_path.rstrip('/')}/{file_name}_mean.png")


    def get_explaination_subgraph(self, explaination):
        return explaination

    def visualize(self, explaination=None, file_name="explaination_graph"):
        append_explainations = True
        if len(self.explainations)!=0: 
            append_explainations = False

        if not explaination: 

            for batch in self.data:
                explainations = self.explain(batch, raw=True)
                subgraph = self.get_explaination_subgraph(explainations)
                 
                self._importance_plot(subgraph, file_name)

                if append_explainations: 
                    self.explainations.update(subgraph)

        else: 
            subgraph = self.get_explaination_subgraph(explaination)
            self._importance_plot(subgraph, file_name)


class GNNExplainerPrune(GlobalGNNExplain): 
    def __init__(self, data_path: str, out_path: str = "explainations/", checkpoint_path: str = None, batch_size: int = 16, test: bool = False, planes=['u', 'v', 'y'], n_batches=None, message_passing_steps=5):
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test, planes, n_batches, message_passing_steps=message_passing_steps)
        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node', 
            return_type="raw")
        
        self.explainer = HeteroExplainer(
            model=self.model, 
            algorithm=NonTrainedHeteroGNNExplainer(epochs=60, plane=self.planes), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="object",
        )

    def explain(self, data):
        explaination = self.explainer(graph=data)
        return explaination

    def get_explaination_subgraph(self, explaination):
        edge_mask = explaination['edge_mask']
        masked_graph = get_masked_graph(
            explaination['graph'], edge_mask=edge_mask, planes=self.planes
        )
        masked_graph['edge_mask'] = edge_mask
        return masked_graph
    