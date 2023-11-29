from nugraph.explain_graph.gnn_explain import GlobalGNNExplain
from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroGNNExplainer, HeteroExplainer
from torch_geometric.explain import ModelConfig
from nugraph.explain_graph.edge_visuals import EdgeVisuals
import matplotlib.pyplot as plt 



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
        figure, subplots = plt.subplots(2, 3, figsize=(16*3, 16*2))
        for index, plane in enumerate(self.planes): 
            subgraph_kx, _, _ = plot_engine.make_subgraph(subgraph, plane=plane)
            node_list = subgraph_kx.nodes 
            subplot = subplots[0, index]
            subplot.set_title(plane)

            plot_engine.plot_graph(subgraph, subgraph_kx, plane, node_list, subplot)

            importance = subgraph['node_mask'][plane].mean(axis=0)
            subplots[1, index].bar(x=range(len(importance)), height=importance)

        plt.savefig(f"{self.out_path.rstrip('/')}/{file_name}.png")

    def get_explaination_subgraph(self, explaination):
        return explaination

    def visualize(self, explaination=None, file_name="explaination_graph"):
        append_explainations = True
        if len(self.explainations)!=0: 
            append_explainations = False

        if not explaination: 

            for index, batch in enumerate(self.data):
                explainations = self.explain(batch, raw=True)
                subgraph = self.get_explaination_subgraph(explainations)
                 
                self._importance_plot(subgraph, file_name)

                if append_explainations: 
                    self.explainations.update(subgraph)

        else: 
            subgraph = self.get_explaination_subgraph(explaination)
            self._importance_plot(subgraph, file_name)
