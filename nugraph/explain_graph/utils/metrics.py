import torch
import torch.nn.functional as F
from torchmetrics.functional import recall


def _get_mask(graph):
    edge_mask = None
    node_mask = None
    if graph.collect("edge_mask") != {}:
        edge_mask = graph.collect("edge_mask")
    if graph.collect("node_mask") != {}:
        node_mask = graph.collect("node_mask")
    return node_mask, edge_mask


def unfaithfulness(explainer, explanation):
    y = explainer.get_prediction(explanation)

    node_mask, edge_mask = _get_mask(explanation)
    y_hat = explainer.get_masked_prediction(
        explanation, edge_mask=edge_mask, node_mask=node_mask
    )

    kl_div = {
        plane: torch.nn.functional.kl_div(
            torch.nn.functional.softmax(y[plane].float(), dim=-1),
            torch.nn.functional.softmax(y_hat[plane].float(), dim=-1),
            reduction="batchmean",
        )
        for plane in explanation.collect("x").keys()
    }
    return {plane: 1 - float(torch.exp(-kl_div[plane])) for plane in kl_div.keys()}


def fidelity(explainer, explanation):
    node_mask, edge_mask = _get_mask(explanation)
    y = explainer.get_prediction(explanation)

    explain_y_hat = explainer.get_masked_prediction(
        explanation,
        edge_mask=edge_mask,
        node_mask=node_mask,
    )

    compliment_edge_mask = (
        None
        if edge_mask is None
        else {key: 1 - edge_mask[key] for key in edge_mask.keys()}
    )
    compliment_node_mask = (
        None
        if node_mask is None
        else {key: 1 - node_mask[key] for key in node_mask.keys()}
    )

    complement_y_hat = explainer.get_masked_prediction(
        explanation, edge_mask=compliment_edge_mask, node_mask=compliment_node_mask
    )

    pos_fidelity = {}
    neg_fidelity = {}
    for plane in explanation.collect("x").keys():
        pad_compliment = torch.nn.functional.pad(
            complement_y_hat[plane],
            (0, len(y[plane]) - len(complement_y_hat[plane])),
            value=None,
        )
        pad_explain = torch.nn.functional.pad(
            explain_y_hat[plane],
            (0, len(y[plane]) - len(explain_y_hat[plane])),
            value=None,
        )
        pos_fidelity[plane] = (
            1.0 - (pad_compliment == y[plane]).float().mean().item() + 10 ** (-6)
        )
        neg_fidelity[plane] = (
            1.0 - (pad_explain == y[plane]).float().mean().item()
        ) + 10 ** (-6)

    return pos_fidelity, neg_fidelity


def loss_difference(explainer, explaination):
    y_true = explaination.collect("y_semantic")
    y_pred_full = explainer.get_prediction(explaination, class_out=False)["x_semantic"]

    node_mask, edge_mask = _get_mask(explaination)
    y_pred_mask = explainer.get_masked_prediction(
        explaination, node_mask=node_mask, edge_mask=edge_mask, class_out=False
    )["x_semantic"]

    return {
        plane: (
            recall_loss(y_true[plane].float(), y_pred_full[plane].float()).item(),
            recall_loss(y_true[plane].float(), y_pred_mask[plane].float()).item(),
        )
        for plane in y_true.keys()
    }


def recall_loss(y, y_pred):
    weight = 1 - recall(
        y_pred,
        y,
        "multiclass",
        num_classes=y_pred.size(1),
        average="none",
        ignore_index=-1,
    )
    ce = F.cross_entropy(y_pred, y.long(), reduction="none", ignore_index=-1)
    loss = weight[y.long()] * ce
    return loss.mean()
