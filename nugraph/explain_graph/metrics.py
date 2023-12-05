import copy 
import torch 

def unfaithfulness(explainer, explanation): 
        
    node_mask = explanation.get('node_mask')
    edge_mask = explanation.get('edge_mask')
    x, edge_index = explanation.x, explanation.edge_index
    kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explainer.get_prediction(x, edge_index, **kwargs)['x_semantic']

    y_hat = explainer.get_masked_prediction(x, edge_index, node_mask,
                                            edge_mask, **kwargs)['x_semantic']

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
    explainer_kwargs = {key: explaionation[key] for key in explaionation._model_args}
    converse_kwargs = copy.deepcopy(explainer_kwargs) # I hate shallow copies I hate shallow copies I have shallow copies. 

    y = explaionation.target
    y_hat = explainer.get_prediction(
            None, None, 
            **explainer_kwargs,
        )
    y_hat = explainer.get_target(y_hat)

    explain_y_hat = explainer.get_masked_prediction(
        None, None, 
        node_mask,
        edge_mask,
        **explainer_kwargs,
    )
    explain_y_hat = explainer.get_target(explain_y_hat)

    complement_y_hat = explainer.get_masked_prediction(
        None, None, 
        {key: 1-node_mask[key] for key in node_mask.keys()},
        {key: 1-edge_mask[key] for key in edge_mask.keys()},
        **converse_kwargs,
    )
    complement_y_hat = explainer.get_target(complement_y_hat)
    pos_fidelity = {}
    neg_fidelity = {}
    for plane in node_mask.keys(): 
        pos_fidelity[plane] = 1. - (complement_y_hat[plane] == y[plane]).float().mean().item()
        neg_fidelity[plane] = 1. - (explain_y_hat[plane] == y[plane]).float().mean().item()

    return pos_fidelity, neg_fidelity