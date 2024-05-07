from typing import Optional
import torch
import copy

from torch_geometric.data.hetero_data import HeteroData


class MaskStrats:
    @staticmethod
    def topk_edges(
        edge_weights: torch.Tensor, nexus_edge_weights: torch.Tensor, k: float = 0.333
    ):
        tokp_edges = torch.topk(
            edge_weights.ravel(), k=int(len(edge_weights.ravel()) * k), dim=0
        ).indices
        tokp_edges_nexus = torch.topk(
            nexus_edge_weights.ravel(),
            k=int(len(nexus_edge_weights.ravel()) * k),
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


def mask_nodes(
    graph: HeteroData,
    node_mask: dict,
    planes: list[str] = ["u", "v", "y"],
    marginalize: bool = True,
    make_nan: bool = True,
):
    new_nodes = {}
    for plane in planes:
        mask = node_mask[plane].sigmoid()
        node_features = graph[plane]["x"][:, :4]

        if make_nan:
            mask[mask < 0.5] = torch.nan
            print(mask)

        if marginalize:
            z = torch.normal(
                mean=torch.zeros_like(node_features, dtype=torch.float) - node_features,
                std=torch.ones_like(node_features, dtype=torch.float) / 2,
            )
            new_nodes[plane] = node_features + z * (1 - mask)

        else:
            new_nodes[plane] = node_features * mask

    new_graph = copy.deepcopy(graph)
    for key, nodes in new_nodes.items():
        new_graph[key]["x"] = nodes

    return new_graph


def mask_edges(
    graph: HeteroData,
    edge_mask: dict,
    planes: list[str] = ["u", "v", "y"],
    mask_strategy=MaskStrats.top_quartile,
):
    keep_edges = {}
    for plane in planes:
        edge_weights = edge_mask[(plane, "plane", plane)].sigmoid()
        nexus_edge_weights = edge_mask[(plane, "nexus", "sp")].sigmoid()

        edges, edges_nexus = mask_strategy(edge_weights, nexus_edge_weights)
        keep_edges[(plane, "nexus", "sp")] = edges_nexus
        keep_edges[(plane, "plane", plane)] = edges

    subgraph = graph.edge_subgraph(keep_edges)
    return subgraph


def get_masked_graph(
    graph: HeteroData,
    edge_mask: Optional[dict] = None,
    node_mask: Optional[dict] = None,
    mask_strategy: MaskStrats = MaskStrats.top_quartile,
    planes: list[str] = ["u", "v", "y"],
    make_nodes_nan: bool = True,
):
    node_mask = node_mask if node_mask != {} else None
    edge_mask = edge_mask if edge_mask != {} else None
    if node_mask is not None:
        graph = mask_nodes(graph, node_mask, planes, make_nan=make_nodes_nan)
    if edge_mask is not None:
        graph = mask_edges(graph, edge_mask, planes, mask_strategy=mask_strategy)

    return graph
