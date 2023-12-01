import torch 
import copy


def get_masked_graph(graph, node_mask, edge_mask, planes=['u', 'v', 'y']): 
    masked_graph = copy.deepcopy(graph)
    for plane in planes: 

        node_weights = node_mask[plane].sigmoid()
        edge_weights = edge_mask[plane].sigmoid()
        nexus_edge_weights = edge_mask[f"{plane}_nexus"].sigmoid()

        topk_nodes = torch.topk(node_weights.ravel(), k=int(len(node_weights.ravel())/3), dim=0)
        tokp_edges = torch.topk(edge_weights.ravel(), k=int(len(edge_weights.ravel())/3), dim=0)
        tokp_edges_nexus = torch.topk(nexus_edge_weights.ravel(), k=int(len(nexus_edge_weights.ravel())/3), dim=0)

        nodes = graph[plane]['x'][topk_nodes.indices]
        edges = graph[(plane, "plane", plane)]['edge_index'][:,tokp_edges.indices]
        edges_nexus = graph[(plane, 'nexus', 'sp')]['edge_index'][:, tokp_edges_nexus.indices]


        masked_graph[(plane, "plane", plane)]['edge_index'] = edges
        #masked_graph[plane]['x'] = nodes
        masked_graph[(plane, 'nexus', 'sp')]['edge_index'] = edges_nexus

    return masked_graph 