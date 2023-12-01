import copy 

def unfaithfulness(explanation): 
        return unfaithfulness

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
        pos_fidelity[plane] = 1. - (complement_y_hat[plane] == y[plane]).float().mean()
        neg_fidelity[plane] = 1. - (explain_y_hat[plane] == y[plane]).float().mean()

    return pos_fidelity, neg_fidelity