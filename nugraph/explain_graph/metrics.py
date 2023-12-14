import copy 
import torch 

def unfaithfulness(explainer, explanation): 
        
    node_mask = explanation.get('node_mask')
    edge_mask = explanation.get('edge_mask')
    graph = explanation['graph']

    y = explainer.get_prediction(graph)['x_semantic']

    y_hat = explainer.get_masked_prediction(graph, node_mask,
                                            edge_mask)['x_semantic']

    kl_div = {
        plane: torch.nn.functional.kl_div(
            torch.nn.functional.softmax(y[plane], dim=-1), 
            torch.nn.functional.softmax(y_hat[plane], dim=-1), reduction='batchmean') 
            for plane in node_mask.keys()
    }
    return {plane: 1 - float(torch.exp(-kl_div[plane])) for plane in node_mask.keys()}

def fidelity(explainer, explaionation): 
    node_mask = explaionation.get('node_mask')
    edge_mask = explaionation.get('edge_mask')
    graph = explaionation['graph']

    y = explaionation.target
    y_hat = explainer.get_prediction(graph)
    y_hat = explainer.get_target(y_hat)

    explain_y_hat = explainer.get_masked_prediction(
        graph,
        node_mask,
        edge_mask,
    )
    explain_y_hat = explainer.get_target(explain_y_hat)

    complement_y_hat = explainer.get_masked_prediction(
        graph,
        {key: 1-node_mask[key] for key in node_mask.keys()},
        {key: 1-edge_mask[key] for key in edge_mask.keys()},
    )
    complement_y_hat = explainer.get_target(complement_y_hat)
    pos_fidelity = {}
    neg_fidelity = {}
    for plane in node_mask.keys(): 
        pad_compliment = torch.nn.functional.pad(complement_y_hat[plane], (0, len(y[plane]) - len(complement_y_hat[plane])), value=None)
        pad_explain = torch.nn.functional.pad(explain_y_hat[plane], (0, len(y[plane]) - len(explain_y_hat[plane])), value=None)
        pos_fidelity[plane] = 1. - (pad_compliment == y[plane]).float().mean().item()
        neg_fidelity[plane] = 1. - (pad_explain == y[plane]).float().mean().item()

    return pos_fidelity, neg_fidelity