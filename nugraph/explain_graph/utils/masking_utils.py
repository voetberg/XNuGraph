import torch 
import copy

from torch_geometric.data.hetero_data import HeteroData


class MaskStrats: 
    @staticmethod
    def topk_edges(graph: HeteroData, edge_weights:torch.Tensor, nexus_edge_weights:torch.Tensor, plane:str): 
        tokp_edges = torch.topk(edge_weights.ravel(), k=int(len(edge_weights.ravel())/3), dim=0)
        tokp_edges_nexus = torch.topk(nexus_edge_weights.ravel(), k=int(len(nexus_edge_weights.ravel())/3), dim=0)
        edges = graph[(plane, "plane", plane)]['edge_index'][:,tokp_edges.indices]
        edges_nexus = graph[(plane, 'nexus', 'sp')]['edge_index'][:, tokp_edges_nexus.indices]

        return edges, edges_nexus

    @staticmethod
    def top_percentile(graph, edge_weights, nexus_edge_weights, plane, percentile): 
        try: 
            edge_index = torch.where(edge_weights>edge_weights.quantile(1-percentile))[0]
            
        except RuntimeError: 
            edge_index = []
        try: 
            edge_nexus_index = torch.where(nexus_edge_weights>nexus_edge_weights.quantile(1-percentile))[0]
        except RuntimeError: 
            edge_nexus_index = []

        edges = graph[(plane, "plane", plane)]['edge_index'][:,edge_index]
        edges_nexus = graph[(plane, 'nexus', 'sp')]['edge_index'][:, edge_nexus_index]
        return edges, edges_nexus


    @staticmethod
    def top_quartile(graph: HeteroData, edge_weights:torch.Tensor, nexus_edge_weights:torch.Tensor, plane:str): 
        return MaskStrats.top_percentile(graph, edge_weights, nexus_edge_weights, plane, percentile=0.25)

    @staticmethod
    def top_tenth(graph: HeteroData, edge_weights:torch.Tensor, nexus_edge_weights:torch.Tensor, plane:str): 
        return MaskStrats.top_percentile(graph, edge_weights, nexus_edge_weights, plane, percentile=0.1)

def get_masked_graph(graph:HeteroData, node_mask:dict, edge_mask:dict, mask_strategy: MaskStrats=MaskStrats.top_quartile, planes:list[str]=['u', 'v', 'y']): 

    masked_graph = copy.deepcopy(graph)
    for plane in planes: 

        if node_mask is not None: 
            node_weights = node_mask[plane].sigmoid()
            nodes = graph[plane]['x']*node_weights
            masked_graph[plane]['x'] = nodes

        if edge_mask is not None: 
            edge_weights = edge_mask[plane].sigmoid()
            nexus_edge_weights = edge_mask[f"{plane}_nexus"].sigmoid()

            edges, edges_nexus = mask_strategy(graph, edge_weights, nexus_edge_weights, plane=plane)

            masked_graph[(plane, "plane", plane)]['edge_index'] = edges
            masked_graph[(plane, 'nexus', 'sp')]['edge_index'] = edges_nexus

    return masked_graph 


def apply_predefined_mask(graph, node_mask, edge_mask, nexus_edge_mask, planes): 
    masked_graph = copy.deepcopy(graph)
    multi_col_keys = ['pos', 'x', 'x_semantic']
    single_col_keys = ['x_filter', 'batch', 'y_instance', 'y_semantic', 'id']

    for plane in planes: 

        nodes = {}
        nodes.update({key: graph[plane][key][node_mask[plane],:] for key in multi_col_keys})
        nodes.update({key: graph[plane][key][node_mask[plane]] for key in single_col_keys})

        old_index = node_mask[plane].nonzero().squeeze().tolist()
        if type(old_index) == int: 
            old_index = [old_index]
        new_index = torch.arange(len(old_index)).tolist()
        index_map = dict(zip(old_index, new_index))

        edges = graph[(plane, "plane", plane)]['edge_index'][:,edge_mask[plane]].cpu().apply_(index_map.get)
        edges_nexus = graph[(plane, 'nexus', 'sp')]['edge_index'][:,nexus_edge_mask[plane]].cpu().apply_(index_map.get)
        
        masked_graph[(plane, "plane", plane)]['edge_index'] = edges
        masked_graph[(plane, 'nexus', 'sp')]['edge_index'] = edges_nexus
        
        for key in nodes.keys(): 
            masked_graph[plane][key] = nodes[key]

    return masked_graph