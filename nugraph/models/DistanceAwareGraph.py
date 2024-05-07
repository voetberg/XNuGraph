# A Version of NuGraph that allows you to remove specific features from X vectors
# without it throwing a hissy-fit about shapes

from torch import Tensor, concatenate, empty
from nugraph.models.NuGraph2 import NuGraph2
from nugraph.models.distance_aware_plane import DistanceAwarePlaneNet


class DistanceAwareGraph(NuGraph2):
    def __init__(
        self,
        in_features: int = 5,
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

        edge_features = 2
        self.plane_net = DistanceAwarePlaneNet(
            in_features,
            planar_features,
            edge_features,
            len(semantic_classes),
            planes,
            checkpoint=checkpoint,
        )

    def forward(
        self,
        x: dict[str, Tensor],
        edge_index_plane: dict[str, Tensor],
        edge_index_nexus: dict[str, Tensor],
        edge_attr_plane: dict[str, Tensor],
        nexus: Tensor,
        batch: dict[str, Tensor],
    ):
        x = {plane: x[plane][:, : self.in_features] for plane in self.planes}
        m = self.encoder(x)
        for _ in range(self.num_iters):
            # shortcut connect features
            for i, p in enumerate(self.planes):
                s = x[p].detach().unsqueeze(1).expand(-1, m[p].size(1), -1)
                m[p] = concatenate((m[p], s), dim=-1)
            print(edge_attr_plane["u"].shape)
            self.plane_net(m, edge_index_plane, edge_attr_plane)
            self.nexus_net(m, edge_index_nexus, nexus)
        ret = {}
        for decoder in self.decoders:
            ret.update(decoder(m, batch))
        return ret

    def unpack_batch(self, batch):
        return (
            batch.collect("x"),
            {p: batch[p, "plane", p].edge_index for p in self.planes},
            {p: batch[p, "nexus", "sp"].edge_index for p in self.planes},
            {p: batch.collect("features")[p, "plane", p] for p in self.planes},
            empty(batch["sp"].num_nodes, 0),
            {p: batch[p].batch for p in self.planes},
        )
