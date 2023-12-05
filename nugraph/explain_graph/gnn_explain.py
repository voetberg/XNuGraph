from nugraph.explain_graph.explain_local import ExplainLocal
from nugraph.explain_graph.edge_visuals import EdgeVisuals

from torch_geometric.explain import ModelConfig
from datetime import datetime 
from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroGNNExplainer, HeteroExplainer
from nugraph.explain_graph.masking_utils import get_masked_graph

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
            EdgeVisuals(planes=self.planes).plot(graph=subgraph, outdir=self.out_path, file_name=f"{file_name}.png")

    def explain(self, data):
        graph = self.process_graph(next(iter(data))) 

        explaination = self.explainer(graph=graph)

        metrics = self.calculate_metrics(explaination)
        self.metrics[str(len(self.metrics))] = metrics 

        return explaination