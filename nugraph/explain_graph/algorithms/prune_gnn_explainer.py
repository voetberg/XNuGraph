from nugraph.explain_graph.algorithms.hetero_gnnexplainer import HeteroGNNExplainer
from nugraph.explain_graph.utils.masking_utils import get_masked_graph
from nugraph.explain_graph.utils.load import Load

from nugraph.models.plane import PlaneNet
from nugraph.models.nexus import NexusNet

from torch_geometric.explain import Explanation

import torch
import copy
import tqdm


class NonTrainedHeteroGNNExplainer(HeteroGNNExplainer):
    def __init__(
        self,
        epochs: int = 50,
        lr: float = 0.01,
        planes=["u", "v", "y"],
        message_passing_steps=5,
        semantic_classes=["MIP", "HIP", "shower", "michel", "diffuse"],
        **kwargs,
    ):
        # After N message passing steps, have the node's values
        # very similar if they are in either the edge-pruned or non-edge-pruned
        super().__init__(epochs, lr, planes, **kwargs)
        self.message_passing_steps = message_passing_steps
        self.semantic_classes = semantic_classes

    def model(self, graph):
        # Apply the forward passing pre-processing used for the graph
        # Just not the weights?
        # This may be a biasing but not? Sure if that's going to be a big deal or not

        class message_passing_model:
            def __init__(self, planes, semantic_classes) -> None:
                self.planes = planes
                self.plane = PlaneNet(
                    in_features=6,
                    planar_features=6,
                    num_classes=len(semantic_classes),
                    planes=planes,
                    checkpoint=False,
                )

                self.nexus = NexusNet(
                    planar_features=6,
                    nexus_features=1,
                    num_classes=len(semantic_classes),
                    planes=planes,
                    checkpoint=False,
                )

            def forward(
                self, x, edge_index_plane, edge_index_nexus, nexus, n_iterations
            ):
                for p in self.planes:
                    s = x[p].detach().unsqueeze(1)
                    x[p] = torch.cat((x[p].unsqueeze(1), s), dim=-1)

                for _ in range(n_iterations):
                    self.plane(x, edge_index_plane)
                    self.nexus(x, edge_index_nexus, nexus)

                    for p in self.planes:
                        # Send things back to the correct size
                        s = x[p].detach()
                        x[p] = torch.cat((x[p], s), dim=-1)

                return x

        m = message_passing_model(self.planes, self.semantic_classes)

        x, edge_index_plane, edge_index_nexus, nexus, _ = Load.unpack(graph)

        x = m.forward(
            x,
            edge_index_plane,
            edge_index_nexus,
            nexus,
            n_iterations=self.message_passing_steps,
        )
        x = torch.cat([x[p] for p in self.planes])
        return x

    def forward(self, model, graph):
        prediction = copy.deepcopy(graph)

        prediction = self._train(model, prediction)

        edge_mask = {
            key: self._post_process_mask(
                self.edge_mask[key],
                self.hard_edge_mask[key],
            )
            for key in self.edge_mask.keys()
        }

        self._clean_model(model)
        explainer = Explanation(edge_mask=edge_mask)

        for plane in self.planes:
            explainer[plane] = {}
            explainer[plane]["pos"] = graph[plane]["pos"]
            # explainer[plane]['pred_label'] = prediction[plane]['x_semantic']
            #  explainer[plane]['sem_label'] = graph[plane]['y_semantic']
            explainer[plane]["x"] = graph[plane]["x"]

        return explainer

    def _train(self, model, graph, node_index=None, loss_history=None, **kwargs):
        copy_graph = copy.deepcopy(graph)
        graph.requires_grad = True

        y = self.model(copy_graph)

        parameters = self.assign_nexus_masks(graph, node=False, edge=True)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in tqdm.tqdm(range(self.epochs)):
            optimizer.zero_grad()
            stepped_graph = get_masked_graph(
                graph, node_mask=None, edge_mask=self.edge_mask
            )
            y_hat = self.model(stepped_graph)

            # Match the output of the model
            assert y.size() == y_hat.size(), print(
                f"{y.size()} vs {y_hat.size()}"
            )  # Personal check that things are not weirdly transposed
            loss = self._loss(y_hat, y)
            self.loss_history.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).

            if i == 0 and self.edge_mask is not None:
                self.hard_edge_mask = {
                    key: self.edge_mask[key].grad != 0 for key in self.edge_mask.keys()
                }

        return stepped_graph

    def _loss(self, y_hat, y):
        _, m_edge = self.get_nexus_m()
        loss = self.loss_function(y_hat, y)

        edge_reduce = getattr(torch, self.coeffs["edge_reduction"])
        loss = loss + self.coeffs["edge_size"] * edge_reduce(m_edge)
        ent = -m_edge * torch.log(m_edge + self.coeffs["EPS"]) - (
            1 - m_edge
        ) * torch.log(1 - m_edge + self.coeffs["EPS"])
        loss = loss + self.coeffs["edge_ent"] * ent.mean()

        return loss

    def loss_function(self, y_hat, y):
        # compute the kl diverangence between the two distribution of nodes:
        # Want to compare each the distribution of probablities for class, so average over the -1 axis
        return torch.nn.functional.kl_div(y, y_hat, reduction="batchmean")
