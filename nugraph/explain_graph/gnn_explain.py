from nugraph.explain_graph.explain_local import ExplainLocal
from nugraph.explain_graph.edge_visuals import EdgeVisuals

from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.data import HeteroData
from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroGNNExplainer, HeteroExplainer

import torch


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
            algorithm=HeteroGNNExplainer(epochs=100, single_plane=False, plane=self.planes), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="object",
            edge_mask_type="object",
        )

    def filter_edges(self, edges, filter): 
        filter = (filter.sigmoid() >= 0.5)
        edges = edges[:,filter]
        return edges

    def get_explaination_subgraph(self, explaination):
        explination_graph = HeteroData()
        for plane in self.planes: 

            # Apply the mask on each plane: 
            # Rules: Edges mask applied to both 
            # Apply the mask to the nodes
                # Remove the nodes from the edge indices if they are not in the nodes post-mask
            
            node_mask_value = explaination['node_mask'][plane].sigmoid().ravel()

            topk_nodes = torch.topk(node_mask_value, k=int(len(node_mask_value)/3), dim=0)


            edge_weights = explaination['edge_mask'][plane].sigmoid()
            tokp_edges = torch.topk(edge_weights.ravel(), k=int(len(edge_weights.ravel())/3), dim=0)

            nodes = explaination[plane]['x'][topk_nodes.indices]
            edges = explaination[(plane, "plane", plane)]['edge_index'][:,tokp_edges.indices]

            assert edges.size(0)==2

            explination_graph[(plane, "plane", plane)]['edge_index'] = edges
            explination_graph[(plane, "plane", plane)]['weight'] = edge_weights

            explination_graph[plane]['node_mask'] = explaination['node_mask'][plane]
            explination_graph[plane]['pos'] = explaination[plane]['pos'] 
            explination_graph[plane]['x'] = nodes

            # TODO: Swap out for the actual labels
            explination_graph[plane]['pred_label'] = explaination[plane]['y_semantic']
            explination_graph[plane]['sem_label'] = explaination[plane]['y_semantic']


        # Make the nexus graph: 
        # get the nodes that have a connection to the nexus - use their weight and pos to show rep in nexus plane
        explination_graph['nexus'] = {}
        for plane in self.planes: 
            ""

        return explination_graph
    
    def visualize(self, explaination=None, file_name="explaination_graph", interactive=False):
        append_explainations = True
        if len(self.explainations)!=0: 
            append_explainations = False

        if not explaination: 

            for index, batch in enumerate(self.data):
                explainations = self.explain(batch, raw=True)
                subgraph = self.get_explaination_subgraph(explainations)
                 
                EdgeVisuals().plot(graph=subgraph, outdir=self.out_path, file_name=f"{file_name}/{index}.png")
                
                if append_explainations: 
                    self.explainations.update(subgraph)

        else: 
            subgraph = self.get_explaination_subgraph(explaination)

            if interactive: 
                [EdgeVisuals(planes=self.planes).interactive_plot(graph=subgraph, plane=plane, outdir=self.out_path, file_name=f"{plane}_{file_name}.html") for plane in self.planes]
            else: 
                EdgeVisuals(planes=self.planes).plot(graph=subgraph, outdir=self.out_path, file_name=f"{file_name}.png")
    
    def explain(self, data, node_index=[8], raw:bool=True):
        graph = self.process_graph(next(iter(data))) 
        explaination = self.explainer(graph=graph)
        if not raw: 
            explaination = self.get_explaination_subgraph(explaination)

        return explaination