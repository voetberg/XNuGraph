import copy 
import torch 

def unfaithfulness(explainer, explanation): 
        
    node_mask = explanation.collect('node_mask')
    edge_mask = explanation.collect('edge_mask')
    y = explainer.get_prediction(explanation)['x_semantic']

    y_hat = explainer.get_masked_prediction(explanation, edge_mask)['x_semantic']

    kl_div = {
        plane: torch.nn.functional.kl_div(
            torch.nn.functional.softmax(y[plane], dim=-1), 
            torch.nn.functional.softmax(y_hat[plane], dim=-1), reduction='batchmean') 
            for plane in node_mask.keys()
    }
    return {plane: 1 - float(torch.exp(-kl_div[plane])) for plane in node_mask.keys()}

def fidelity(explainer, explanation): 
    node_mask = explanation.collect('node_mask')
    edge_mask = explanation.collect('edge_mask')
    
    y = explanation.target
    y_hat = explainer.get_prediction(explanation)
    y_hat = explainer.get_target(y_hat)

    explain_y_hat = explainer.get_masked_prediction(
        explanation,
        edge_mask,
    )
    explain_y_hat = explainer.get_target(explain_y_hat)

    complement_y_hat = explainer.get_masked_prediction(
        explanation,
        {key: 1-edge_mask[key] for key in edge_mask.keys()},
    )
    complement_y_hat = explainer.get_target(complement_y_hat)
    pos_fidelity = {}
    neg_fidelity = {}
    for plane in node_mask.keys(): 
        pad_compliment = torch.nn.functional.pad(complement_y_hat[plane], (0, len(y[plane]) - len(complement_y_hat[plane])), value=None)
        pad_explain = torch.nn.functional.pad(explain_y_hat[plane], (0, len(y[plane]) - len(explain_y_hat[plane])), value=None)
        pos_fidelity[plane] = 1. - (pad_compliment == y[plane]).float().mean().item() + 10**(-6)
        neg_fidelity[plane] = (1. - (pad_explain == y[plane]).float().mean().item()) + 10**(-6)

    return pos_fidelity, neg_fidelity