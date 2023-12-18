from nugraph.explain_graph.explain import ExplainLocal
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals

from torch_geometric.explain import ModelConfig
from datetime import datetime 
from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroGNNExplainer, HeteroExplainer
from nugraph.explain_graph.algorithms.multi_edge_hetero_gnnexplainer import MultiEdgeHeteroGNNExplainer
from nugraph.explain_graph.utils.masking_utils import get_masked_graph
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals, InteractiveEdgeVisuals, make_subgraph_kx

import matplotlib.pyplot as plt 

class GlobalGNNExplain(ExplainLocal): 
    def __init__(self, data_path: str, out_path: str = "explainations/", checkpoint_path: str = None, batch_size: int = 16, test: bool = False, planes=['u', 'v', 'y']):
        self.planes = planes
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test)

        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node', 
            return_type="raw")
        
        self.explainer = HeteroExplainer(
            model=self.model, 
            algorithm=HeteroGNNExplainer(epochs=300, plane=self.planes), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="object",
            edge_mask_type="object",
        )

    def get_explaination_subgraph(self, explaination):
        node_mask = explaination['node_mask']
        edge_mask = explaination['edge_mask']
        return get_masked_graph(
            explaination['graph'], node_mask, edge_mask, planes=self.planes
        )
    
    def visualize(self, explaination, file_name=None, interactive=False):

        if file_name is None: 
            file_name = f"subgraph_{datetime.now().timestamp()}"

        subgraph = self.get_explaination_subgraph(explaination)

        if interactive: 
            [EdgeVisuals(planes=self.planes).interactive_plot(graph=subgraph, plane=plane, outdir=self.out_path, file_name=f"{plane}_{file_name}.html") for plane in self.planes]
        else: 
            EdgeVisuals(planes=self.planes).plot(graph=subgraph, outdir=self.out_path, file_name=f"{file_name}.png", nexus_distribution=True)

    def explain(self, data):
        graph = self.process_graph(next(iter(data))) 

        explaination = self.explainer(graph=graph)

        metrics = self.calculate_metrics(explaination)
        self.metrics[str(len(self.metrics))] = metrics 

        return explaination
    
class ClasswiseGNNExplain(GlobalGNNExplain): 
    def __init__(self, data_path: str, out_path: str = "explainations/", checkpoint_path: str = None, batch_size: int = 16, test: bool = False, planes=['u', 'v', 'y']):
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test, planes)
        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node', 
            return_type="raw")
        
        self.explainer = HeteroExplainer(
            model=self.model, 
            algorithm=MultiEdgeHeteroGNNExplainer(epochs=300, plane=self.planes), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="object",
            edge_mask_type="object",
        )

    def visualize(self, explaination, file_name=None, interactive=False):
        for key in explaination.keys(): 
            class_file_name = file_name+str(key)
            graph = explaination[key]
            graph.graph.node_mask = graph.node_mask 
            graph.graph.edge_mask = graph.edge_mask
            super().visualize(explaination[key], class_file_name, interactive)
        
        self.explainer.algorithm.plot_loss(
            f"{self.out_path.rstrip('/')}/explainer_loss.png"
        )
    
    def calculate_metrics(self, explainations):
        metrics = {}
        for key in explainations: 
            metrics[key] = super().calculate_metrics(explainations[key])
        return metrics
    
class GNNExplainFeatures(GlobalGNNExplain): 
    def __init__(self, data_path: str, out_path: str = "explainations/", checkpoint_path: str = None, batch_size: int = 16, test: bool = False, planes=['u', 'v', 'y']):
        self.planes = planes
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test)

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
