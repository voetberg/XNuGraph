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
                explainations = self.explain(batch, raw=True)
                for key in explainations.keys(): 
                    explainations[key].visualize_graph(f"{file_name}/{index}_plane_{key}.png")
                
                if append_explainations: 
                    self.explainations.update(explaination.get_explanation_subgraph())

        else: 
            assert file_name is not None, "Please supply a file name"
            explaination.visualize_graph(f"{self.out_path}/{file_name}")

    
    def explain(self, data, raw:bool=True):
        plane_explain = {}
        for plane in self.planes: 
            explaination = self.explainer(data,  plane=plane)
            if not raw: 
                explaination = explaination.get_explanation_subgraph()
            plane_explain[plane] = explaination

        return plane_explain 