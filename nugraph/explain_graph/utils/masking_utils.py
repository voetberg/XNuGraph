from typing import Optional
import torch
import copy

from torch_geometric.data.hetero_data import HeteroData


class MaskStrats:
    @staticmethod
    def topk_edges(edge_weights: torch.Tensor, nexus_edge_weights: torch.Tensor):
        tokp_edges = torch.topk(
            edge_weights.ravel(), k=int(len(edge_weights.ravel()) / 3), dim=0
        ).indices
        tokp_edges_nexus = torch.topk(
            nexus_edge_weights.ravel(),
            k=int(len(nexus_edge_weights.ravel()) / 3),
            dim=0,
        ).indices
        return tokp_edges, tokp_edges_nexus

    @staticmethod
    def top_percentile(edge_weights, nexus_edge_weights, percentile):
        try:
            edge_index = torch.where(
                edge_weights > edge_weights.quantile(1 - percentile)
            )[0]

        except RuntimeError:
            edge_index = torch.where(nexus_edge_weights == nexus_edge_weights)[0]
        try:
            edge_nexus_index = torch.where(
                nexus_edge_weights > nexus_edge_weights.quantile(1 - percentile)
            )[0]
        except RuntimeError:
            edge_nexus_index = torch.where(nexus_edge_weights == nexus_edge_weights)[0]

        return edge_index, edge_nexus_index

    @staticmethod
    def top_quartile(edge_weights: torch.Tensor, nexus_edge_weights: torch.Tensor):
        return MaskStrats.top_percentile(
            edge_weights, nexus_edge_weights, percentile=0.25
        )

    @staticmethod
    def top_tenth(
        edge_weights: torch.Tensor,
        nexus_edge_weights: torch.Tensor,
    ):
        return MaskStrats.top_percentile(
            edge_weights, nexus_edge_weights, percentile=0.1
        )


def get_masked_graph(
    graph: HeteroData,
    edge_mask: Optional[dict] = None,
    node_mask: Optional[dict] = None,
    mask_strategy: MaskStrats = MaskStrats.top_quartile,
    planes: list[str] = ["u", "v", "y"],
):
    node_mask = node_mask if node_mask != {} else None
    edge_mask = edge_mask if edge_mask != {} else None

    keep_edges = {}
    new_nodes = {}
    for plane in planes:
        if node_mask is not None:
            new_nodes[plane] = graph[plane]["x"][:, :4] * node_mask[plane].sigmoid()

        if edge_mask is not None:
            edge_weights = edge_mask[(plane, "plane", plane)].sigmoid()
            nexus_edge_weights = edge_mask[(plane, "nexus", "sp")].sigmoid()

            edges, edges_nexus = mask_strategy(edge_weights, nexus_edge_weights)
            keep_edges[(plane, "nexus", "sp")] = edges_nexus
            keep_edges[(plane, "plane", plane)] = edges

    if edge_mask is not None:
        subgraph = graph.edge_subgraph(keep_edges)
    else:
        subgraph = copy.deepcopy(graph)

    for key, nodes in new_nodes.items():
        subgraph[key]["x"] = nodes
    return subgraph


def apply_predefined_mask(graph, node_mask, edge_mask, nexus_edge_mask, planes):
    masked_graph = copy.deepcopy(graph)
    multi_col_keys = ["pos", "x", "x_semantic"]
    single_col_keys = ["x_filter", "batch", "y_instance", "y_semantic", "id"]

    for plane in planes:
        nodes = {}
        nodes.update(
            {key: graph[plane][key][node_mask[plane], :] for key in multi_col_keys}
        )
        nodes.update(
            {key: graph[plane][key][node_mask[plane]] for key in single_col_keys}
        )

        old_index = node_mask[plane].nonzero().squeeze().tolist()
        if isinstance(old_index, int):
            old_index = [old_index]
        new_index = torch.arange(len(old_index)).tolist()
        index_map = dict(zip(old_index, new_index))

        edges = (
            graph[(plane, "plane", plane)]["edge_index"][:, edge_mask[plane]]
            .cpu()
            .apply_(index_map.get)
        )
        edges_nexus = (
            graph[(plane, "nexus", "sp")]["edge_index"][:, nexus_edge_mask[plane]]
            .cpu()
            .apply_(index_map.get)
        )

        masked_graph[(plane, "plane", plane)]["edge_index"] = edges
        masked_graph[(plane, "nexus", "sp")]["edge_index"] = edges_nexus

        for key in nodes.keys():
            masked_graph[plane][key] = nodes[key]

    return masked_graph
