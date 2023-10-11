from nugraph.explain_graph.explain_local import ExplainLocal
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

import os 

class GNNExplain(ExplainLocal): 
    def __init__(self, data_path: str, out_path: str = "explainations/", checkpoint_path: str = None, batch_size: int = 16):
        super().__init__(data_path, out_path, checkpoint_path, batch_size)
        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node',
            return_type='probs')
        
        self.explainer = Explainer(
            model=self.model, 
            algorithm=GNNExplainer(epochs=10), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="attributes",
            edge_mask_type="object",
        )

    def visualize(self, explaination=None, file_name=None):
        if not explaination: 
            file_name = f"{self.out_path}/plots"
            if not os.path.exists(file_name):
                os.makedirs(file_name)

            for index, batch in enumerate(self.data):
                explaination = self.explain(batch, raw=True)
                explaination.visualize_graph(f"{file_name}/{index}.png")
        else: 
            assert file_name is not None, "Please supply a file name"
            explaination.visualize_graph(f"{self.out_path}/{file_name}")

    
    def explain(self, data, raw:bool=True, **kwds):
        x = data.x
        edge_index = data.edge_index
        if hasattr(kwds, "node_index"): 
            explaination = self.explainer(x, edge_index, kwds.node_index)
        else: 
            explaination = self.explainer(x, edge_index)

        if not raw: 
            explaination = explaination.get_explanation_subgraph()

        return explaination 