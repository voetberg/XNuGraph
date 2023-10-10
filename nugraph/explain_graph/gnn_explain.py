from nugraph.explain_graph.explain_local import ExplainLocal
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

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
            node_mask_type="object"
        )

    def visualize(self, *args, **kwrds):
        pass 
    
    def explain(self, data, **kwds):
        x = data
        edge_index = data 
        target = data
        explaination = self.explainer(x, edge_index, target).available_explanations
        return explaination 