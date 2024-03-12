import copy
import tqdm

import math
import torch
from torch import Tensor

from torch_geometric.explain import Explainer, ExplainerAlgorithm, GNNExplainer
from torch_geometric.data import HeteroData
from torch_geometric.explain.config import MaskType

from nugraph.explain_graph.utils.load import Load
from nugraph.explain_graph.utils.masking_utils import get_masked_graph
from nugraph.util.RecallLoss import RecallLoss


class HeteroExplainer(Explainer):
    def __init__(
        self,
        model: torch.nn.Module,
        algorithm: ExplainerAlgorithm,
        explanation_type,
        model_config,
        edge_mask_type=None,
        node_mask_type=None,
        threshold_config=None,
        nexus=True,
    ):
        super().__init__(
            model,
            algorithm,
            explanation_type,
            model_config,
            node_mask_type,
            edge_mask_type,
            threshold_config,
        )
        self.nexus = nexus

    def get_prediction(self, graph, class_out=True) -> Tensor:
        x, plane_edge, nexus_edge, nexus, batch = Load.unpack(graph)
        prediction = self.model(x, plane_edge, nexus_edge, nexus, batch)
        if class_out:
            prediction = self.get_target(prediction)
        return prediction

    def get_target(self, prediction: HeteroData) -> Tensor:
        preds = prediction["x_semantic"]
        filter = prediction["x_filter"]

        target = {}
        for plane in preds.keys():
            semantic_prediction = torch.argmax(preds[plane].detach(), dim=-1)
            filter_prediction = filter[plane].detach() < 0.5
            semantic_prediction[filter_prediction] = 6
            target[plane] = semantic_prediction
        return target

    def get_masked_prediction(
        self, graph, edge_mask, node_mask=None, class_out=True
    ) -> Tensor:
        masked_graph = get_masked_graph(graph, edge_mask=edge_mask, node_mask=node_mask)
        out = self.get_prediction(graph=masked_graph, class_out=class_out)
        return out

    def __call__(self, graph):
        target = self.get_prediction(graph)

        self.model.eval()

        explanation = self.algorithm(self.model, graph)
        try:
            explanation.target = target
        except AttributeError:
            for key in explanation:
                explanation[key].target = target

        return explanation


class HeteroGNNExplainer(GNNExplainer):
    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        planes=["u", "v", "y"],
        nexus=True,
        **kwargs,
    ):
        super().__init__(epochs, lr, **kwargs)
        self.planes = planes
        self.loss_history = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nexus = nexus

    def forward(self, model, graph):
        prediction = copy.deepcopy(graph).to(self.device)
        prediction = self._train(model, prediction)

        if self.node_mask is not None:
            node_mask = {
                key: self._post_process_mask(
                    self.node_mask[key],
                    self.hard_node_mask[key],
                )
                for key in self.node_mask.keys()
            }

        if self.edge_mask is not None:
            edge_mask = {
                key: self._post_process_mask(
                    self.edge_mask[key],
                    self.hard_edge_mask[key],
                )
                for key in self.edge_mask.keys()
            }

        for plane in self.planes:
            if self.edge_mask is not None:
                graph[plane, plane].edge_mask = edge_mask[(plane, "plane", plane)]
                graph[plane, "sp"].edge_mask = edge_mask[(plane, "nexus", "sp")]

            if self.node_mask is not None:
                graph[plane].node_mask = node_mask[plane]

        self._clean_model(model)
        return graph

    def assign_planar_masks(self, graph, plane, edge=True, node=True):
        node_mask_type = self.explainer_config.node_mask_type
        x_mask = graph[plane]["x"]
        edge_index_mask = graph[plane, "plane", plane]["edge_index"].to(torch.float)
        edge_mask = None

        (N, F), E = x_mask.size(), edge_index_mask.size(1)
        node_mask = None

        if node:
            if node_mask_type == MaskType.object:
                node_mask = torch.tensor(
                    torch.randn(N, 1) * 0.1, device=self.device, requires_grad=True
                )
            elif node_mask_type == MaskType.attributes:
                node_mask = torch.tensor(
                    torch.randn(N, F) * 0.1, device=self.device, requires_grad=True
                )
        if edge:
            std = torch.nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (2 * N + 10 ** (-3))
            )
            edge_mask = torch.tensor(
                torch.randn(E) * std, device=self.device, requires_grad=True
            )

        parameters = []
        if node_mask is not None:
            parameters.append(node_mask)

        if edge_mask is not None:
            parameters.append(edge_mask)

        return parameters, node_mask, edge_mask

    def assign_nexus_masks(self, graph, edge=True, node=True):
        if node:
            self.node_mask = {}
            self.hard_node_mask = {}

        if edge:
            self.edge_mask = {}
            self.hard_edge_mask = {}

        all_parameters = []

        for plane in self.planes:
            ## planar masks
            parameters, node_mask, edge_mask = self.assign_planar_masks(
                graph, plane, edge=edge, node=node
            )
            all_parameters += parameters

            if node:
                self.node_mask[plane] = node_mask
                self.hard_node_mask[plane] = torch.ones_like(node_mask).to(bool)

            if edge:
                self.edge_mask[(plane, "plane", plane)] = edge_mask
                self.hard_edge_mask[(plane, "plane", plane)] = torch.ones_like(
                    edge_mask
                ).to(bool)

                ## nexus masks
                nexus_edge = graph[(plane, "nexus", "sp")]["edge_index"].to(torch.float)
                N, _ = graph[plane]["x"].size()
                E = nexus_edge.size(1)
                std = torch.nn.init.calculate_gain("relu") * math.sqrt(
                    2.0 / (2 * N + 10 ** (-3))
                )
                edge_mask = torch.tensor(
                    torch.randn(E) * std, device=self.device, requires_grad=True
                )

                self.edge_mask[(plane, "nexus", "sp")] = edge_mask
                self.hard_edge_mask[(plane, "nexus", "sp")] = torch.ones_like(
                    edge_mask
                ).to(bool)

                all_parameters.append(edge_mask)

        return all_parameters

    def assign_masks(self, graph):
        node = self.explainer_config.node_mask_type is not None
        edge = self.explainer_config.edge_mask_type is not None

        if self.nexus:
            return self.assign_nexus_masks(graph, edge, node)
        else:
            if node:
                self.node_mask = {}
                self.hard_node_mask = {}

            if edge:
                self.edge_mask = {}
                self.hard_edge_mask = {}
            params = []
            for plane in self.planes:
                parameters, node_mask, edge_mask = self.assign_planar_masks(
                    graph, plane, edge, node
                )
                params += parameters

                if node_mask is not None:
                    self.node_mask[plane] = node_mask
                if edge_mask is not None:
                    self.edge_mask[plane] = edge_mask

            return params

    def _train(self, model, graph, node_index=None, loss_history=None, **kwargs):
        copy_graph = copy.deepcopy(graph)
        graph.requires_grad = True
        model.step(graph.to(self.device))  # Set the y

        y = (
            torch.concat([t for t in graph.collect("x_semantic").values()])
            .float()
            .to(self.device)
        )

        parameters = self.assign_masks(graph)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in tqdm.tqdm(range(self.epochs)):
            optimizer.zero_grad()
            stepped_graph = get_masked_graph(
                copy_graph, edge_mask=self.edge_mask, node_mask=self.node_mask
            ).to(self.device)

            model.step(stepped_graph)  # Set the y
            y_hat = (
                torch.concat([t for t in stepped_graph.collect("x_semantic").values()])
                .float()
                .to(self.device)
            )

            # Match the output of the model

            assert y.size() == y_hat.size(), print(
                f"{y.size()} vs {y_hat.size()}"
            )  ## Personal check that things are not weirdly transposed

            if node_index is not None:
                y = y[node_index]
                y_hat = y_hat[node_index]

            loss = self._loss(y_hat, y)

            if loss_history is not None:
                loss_history.append(loss.item())

            else:
                self.loss_history.append(loss.item())

            loss.backward(retain_graph=True)
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).

            if i == 0 and self.node_mask is not None:
                self.hard_node_mask = {
                    key: self.node_mask[key].grad != 0 for key in self.node_mask.keys()
                }

            if i == 0 and self.edge_mask is not None:
                self.hard_edge_mask = {
                    key: self.edge_mask[key].grad != 0 for key in self.edge_mask.keys()
                }

        if loss_history is not None:
            return stepped_graph, loss_history

        return stepped_graph

    def get_nexus_m(self):
        m_edge = None
        m_node = None
        if self.node_mask is not None:
            m_node = torch.concat(
                [self.node_mask[key].sigmoid() for key in self.node_mask.keys()]
            )

        if self.edge_mask is not None:
            m_edge = torch.concat(
                [self.edge_mask[key].sigmoid() for key in self.edge_mask.keys()]
            )

        return m_node, m_edge

    def _classification_loss(self, y_hat, y):
        return RecallLoss(ignore_index=-1)(y, y_hat)

    def _loss(self, y_hat, y):
        loss = self._classification_loss(y_hat, y)
        m_node, m_edge = self.get_nexus_m()

        if self.edge_mask is not None:
            edge_reduce = getattr(torch, self.coeffs["edge_reduction"])
            loss = loss + self.coeffs["edge_size"] * edge_reduce(m_edge)
            ent = -m_edge * torch.log(m_edge + self.coeffs["EPS"]) - (
                1 - m_edge
            ) * torch.log(1 - m_edge + self.coeffs["EPS"])
            loss = loss + self.coeffs["edge_ent"] * ent.mean()

        if self.node_mask is not None:
            node_reduce = getattr(torch, self.coeffs["node_feat_reduction"])
            loss = loss + self.coeffs["node_feat_size"] * node_reduce(m_node)
            ent = -m_node * torch.log(m_node + self.coeffs["EPS"]) - (
                1 - m_node
            ) * torch.log(1 - m_node + self.coeffs["EPS"])
            loss = loss + self.coeffs["node_feat_ent"] * ent.mean()
        return loss

    def plot_loss(self, file_name):
        import matplotlib.pyplot as plt

        plt.close("all")

        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Entropy Loss")

        plt.savefig(file_name)
        plt.close("all")
