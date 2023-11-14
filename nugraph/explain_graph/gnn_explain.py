from nugraph.explain_graph.explain_local import ExplainLocal
from nugraph.explain_graph.edge_visuals import EdgeVisuals

from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.data import HeteroData
from nugraph.explain_graph.algorithms.hetero_gnnexplaner import HeteroGNNExplainer, HeteroExplainer

import torch

class PlanarGNNExplain(ExplainLocal): 
    def __init__(self, 
                 data_path: str, 
                 out_path: str = "explainations/", 
                 checkpoint_path: str = None,
                 batch_size: int = 16, 
                 test:bool = False,
                 planes:list=['u', 'v', 'y'], 
                 threshold:float=0.25):
    
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test=test)
        self.planes = planes
        self.threshold = threshold
        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node', 
            return_type="raw")
        self.explainers = [HeteroExplainer(
            model=self.model, 
            algorithm=HeteroGNNExplainer(epochs=100, plane=plane), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="attributes",
            edge_mask_type="object",
        ) for plane in self.planes]


    def get_explaination_subgraph(self, explaination): 
        explination_graph = HeteroData()
        for plane, explain in zip(self.planes, explaination): 

            # Apply the mask on each plane: 
            # Rules: Edges mask applied to both 
            # Apply the mask to the nodes
                # Remove the nodes from the edge indices if they are not in the nodes post-mask
            
            node_mask = explain['node_mask'].sigmoid()

            edge_weights = explain['edge_mask'].sigmoid()

            nodes = (explain[plane]['x']*node_mask.to(float))[:,0].flatten().nonzero()
            edges = explain[(plane, "plane", plane)]['edge_index']

            # Confusing conditional 
            # Only include an edge if both of the nodes included in the edge is not in node mask
            edge_mask = torch.logical_and(torch.isin(edges[0], nodes), torch.isin(edges[1], nodes))
            edge_weight_mask = edge_weights.sigmoid()
            edge_weights = edge_weight_mask*edge_mask.to(int)
            

            assert edges.size(0)==2

            explination_graph[(plane, "plane", plane)]['edge_index'] = edges
            explination_graph[(plane, "plane", plane)]['weight'] = edge_weights

            explination_graph[plane]['node_mask'] = explain['node_mask']
            explination_graph[plane]['pos'] = explain[plane]['pos']  
            explination_graph[plane]['x'] = explain[plane]['x']

            # TODO: Swap out for the actual labels
            explination_graph[plane]['pred_label'] = explain[plane]['y_semantic']
            explination_graph[plane]['sem_label'] = explain[plane]['y_semantic']
            
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
        explainations = []
        for explainer in self.explainers: 
            explaination = explainer(graph=data)
            if not raw: 
                explaination = self.get_explaination_subgraph(explainations)

            explainations.append(explaination)
        return explainations


class GlobalGNNExplain(PlanarGNNExplain): 
    def __init__(self, data_path: str, out_path: str = "explainations/", checkpoint_path: str = None, batch_size: int = 16, test: bool = False, planes=['u', 'v', 'y']):
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test, planes=planes, threshold=.25)

        model_config =  ModelConfig(
            mode='multiclass_classification',
            task_level='node', 
            return_type="raw")
        self.explainers = [HeteroExplainer(
            model=self.model, 
            algorithm=HeteroGNNExplainer(epochs=100, single_plane=False, plane=self.planes), 
            explanation_type='model', 
            model_config=model_config,
            node_mask_type="attributes",
            edge_mask_type="object",
        )]

    def get_explaination_subgraph(self, explaination):
        explaination = explaination[0]

        # Isolate the masks
        node_masks = ""
        edge_masks = ""
        
        ## Apply the planar masks 


        ## Apply the nexus masks

    def visualize(self, explaination=None, file_name="explaination_graph", interactive=False):
        return super().visualize(explaination, file_name, interactive)

        ## TODO         
    
