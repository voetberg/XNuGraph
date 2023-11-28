import pytest 
from nugraph.explain_graph.explain_network import ExplainMessages, ExplainNetwork

def test_init_abstract_defaults(): 
    explain = ExplainNetwork()
    assert hasattr(explain.data)
    assert hasattr(explain.model)

    assert explain.semantic_classes == ['MIP','HIP','shower','michel','diffuse']
    assert explain.planes == ['u', 'v', 'y']

def test_make_linear_decoder():
    e = ExplainNetwork()
    encoder = e.model.encoder.net
    decoder = e.linear_decoder(in_shape=encoder[e.planes[0]][0].weight.shape[0])
    
    out = decoder.forward(encoder.forward(e.data)) 
    assert out is not None 

def test_all_static_decoders(): 
    e = ExplainNetwork()
    decoder_out = e.step_network()
    
    assert None not in decoder_out
    