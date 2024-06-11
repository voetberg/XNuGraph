import copy
import tqdm

import torch

from nugraph.explain_graph.algorithms.hetero_gnnexplainer import HeteroGNNExplainer
from nugraph.explain_graph.utils.masking_utils import get_masked_graph
from nugraph.util.RecallLoss import RecallLoss


class CorrectGNNExplainer(HeteroGNNExplainer):
    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        planes=["u", "v", "y"],
        nexus=True,
        **kwargs,
    ):
        super().__init__(epochs, lr, planes, nexus, **kwargs)

    def _train(self, model, graph, node_index=None, loss_history=None, **kwargs):
        copy_graph = copy.deepcopy(graph)
        graph.requires_grad = True

        y = {
            key: graph.collect("y_semantic")[key].long().to(self.device)
            for key in self.planes
        }
        parameters = self.assign_masks(graph)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in tqdm.tqdm(range(self.epochs)):
            optimizer.zero_grad()
            stepped_graph = get_masked_graph(
                copy_graph, edge_mask=self.edge_mask, node_mask=self.node_mask
            ).to(self.device)

            model.step(stepped_graph)  # Set the y_hat
            y_hat = {
                key: stepped_graph.collect("x_semantic")[key].float().to(self.device)
                for key in self.planes
            }

            if node_index is not None:
                assert y_hat.keys() == node_index.keys()
                y_compare = torch.concat([y[key][node_index[key]] for key in y.keys()])
                y_hat = torch.concat(
                    [y_hat[key][node_index[key]] for key in y_hat.keys()]
                )
            else:
                y_compare = torch.concat([y[key] for key in y_hat.keys()])
                y_hat = torch.concat([y_hat[key] for key in y_hat.keys()])

            loss, recall_loss = self._loss(y_hat, y_compare)
            self.loss_history.append(recall_loss.item())

            loss.backward(retain_graph=True)
            optimizer.step()

            if i == 0 and self.node_mask is not None:
                self.hard_node_mask = {
                    key: self.node_mask[key].grad != 0 for key in self.node_mask.keys()
                }

            if i == 0 and self.edge_mask is not None:
                self.hard_edge_mask = {
                    key: self.edge_mask[key].grad != 0 for key in self.edge_mask.keys()
                }

        return stepped_graph

    def _loss(self, y_hat, y):
        recall_loss = RecallLoss(ignore_index=-1)(y_hat, y)
        loss = copy.copy(recall_loss)
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

        return loss, recall_loss
