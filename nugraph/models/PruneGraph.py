# A Version of NuGraph that allows you to remove specific features from X vectors
# without it throwing a hissy-fit about shapes

from typing import Union
from torch import Tensor, concatenate
from torch.nn import Parameter
from nugraph.models.NuGraph2 import NuGraph2


class PrunedNuGraph(NuGraph2):
    def __init__(
        self,
        prune_feature_index: Union[int, list] = None,
        in_features: int = 4,
        planar_features: int = 8,
        nexus_features: int = 8,
        vertex_features: int = 32,
        planes: list[str] = ["u", "v", "y"],
        semantic_classes: list[str] = ["MIP", "HIP", "shower", "michel", "diffuse"],
        event_classes: list[str] = ["numu", "nue", "nc"],
        num_iters: int = 5,
        event_head: bool = True,
        semantic_head: bool = True,
        filter_head: bool = False,
        vertex_head: bool = False,
        checkpoint: bool = False,
        lr: float = 0.001,
    ):
        super().__init__(
            in_features,
            planar_features,
            nexus_features,
            vertex_features,
            planes,
            semantic_classes,
            event_classes,
            num_iters,
            event_head,
            semantic_head,
            filter_head,
            vertex_head,
            checkpoint,
            lr,
        )
        self.in_features = in_features
        self.planar_features = planar_features
        self.nexus_features = nexus_features

        self.prune_feature = (
            prune_feature_index
            if isinstance(prune_feature_index, list)
            else [prune_feature_index]
        )
        self.include_index = [
            feature
            for feature in range(self.in_features)
            if feature not in prune_feature_index
        ]

    def forward(
        self,
        x: dict[str, Tensor],
        edge_index_plane: dict[str, Tensor],
        edge_index_nexus: dict[str, Tensor],
        nexus: Tensor,
        batch: dict[str, Tensor],
    ):
        x = {plane: x[plane][:, self.include_index] for plane in self.planes}
        m = self.encoder_forward(x)
        self.planar_reshape()

        for _ in range(self.num_iters):
            # shortcut connect features
            for plane in self.planes:
                s = x[plane].detach().unsqueeze(1).expand(-1, m[plane].size(1), -1)
                m[plane] = concatenate((m[plane], s), dim=-1)
            self.plane_net(m, edge_index_plane)
            self.nexus_net(m, edge_index_nexus, nexus)
        ret = {}
        for decoder in self.decoders:
            ret.update(decoder(m, batch))
        return ret

    def encoder_forward(self, x):
        for plane in self.planes:
            for module in self.encoder.net[plane][0].net:
                module.weight = Parameter(module.weight[:, self.include_index])

        return self.encoder(x)

    def planar_reshape(self):
        current_shape_edge = (
            self.plane_net.net[self.planes[0]].edge_net[0].net[0].weight.shape[-1]
        )

        if current_shape_edge == 2 * (self.planar_features + self.in_features):
            remove_weights = []
            for feature in self.prune_feature:
                remove_weights.append(self.planar_features + feature)
                remove_weights.append(2 * remove_weights[-1])

            for plane in self.planes:
                for module in self.plane_net.net[plane].edge_net[0].net:
                    edge_shape = module.weight.shape[-1]
                    include = [
                        index
                        for index in range(edge_shape)
                        if index not in remove_weights
                    ]
                    module.weight = Parameter(module.weight[:, include])

                for module in self.plane_net.net[plane].node_net[0].net:
                    node_shape = module.weight.shape[-1]
                    include = [
                        index
                        for index in range(node_shape)
                        if index not in remove_weights
                    ]
                    module.weight = Parameter(module.weight[:, include])
