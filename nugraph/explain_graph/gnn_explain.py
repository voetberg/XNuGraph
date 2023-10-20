from nugraph.explain_graph.explain_local import ExplainLocal
from torch_geometric.explain import Explainer, ModelConfig
from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroGNNExplainer
import os 

class GNNExplain(ExplainLocal): 
    def __init__(self, 
                 data_path: str, 
                 out_path: str = "explainations/", 
                 checkpoint_path: str = None,
                 batch_size: int = 16, 
                 planes=['u', 'v', 'y']):
        super().__init__(data_path, out_path, checkpoint_path, batch_size)
        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node')
        
        self.explainer = Explainer(
            model=self.model, 
            algorithm=HeteroGNNExplainer(epochs=10), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="attributes",
            edge_mask_type="object",
        )
        self.planes = planes

    def visualize(self, explaination=None, file_name=None):
        append_explainations = True
        if len(self.explainations)!=0: 
            append_explainations = False

        if not explaination: 
            file_name = f"{self.out_path}/plots"
            if not os.path.exists(file_name):
                os.makedirs(file_name)

            for index, batch in enumerate(self.data):
                explaination = self.explain(batch, raw=True)
                explaination.visualize_graph(f"{file_name}/{index}.png")
                
                if append_explainations: 
                    self.explainations.update(explaination.get_explanation_subgraph())

        else: 
            assert file_name is not None, "Please supply a file name"
            explaination.visualize_graph(f"{self.out_path}/{file_name}")

    
    def explain(self, data, raw:bool=True, **kwds):
        x, edge_plane, edge_nexus, nexus, batch = self.unpack(data)
        plane_explain = {}
        for plane in self.planes: 
            if hasattr(kwds, "node_index"): 
                explaination = self.explainer(x, edge_plane, kwds.node_index, edge_index_nexus=edge_nexus, plane=plane, nexus=nexus, batch=batch)
            else: 
                explaination = self.explainer(x, edge_plane, edge_index_nexus=edge_nexus, nexus=nexus, batch=batch, plane=plane)

            if not raw: 
                explaination = explaination.get_explanation_subgraph()
            plane_explain[plane] = explaination

        return plane_explain 