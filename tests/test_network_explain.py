import pytest 
from nugraph.explain_graph.explain_network import ExplainMessages, ExplainNetwork

def test_init_abstract_defaults(): 
    explain = ExplainNetwork(test=True)
    assert hasattr(explain, 'data')
    assert hasattr(explain, 'model')

    assert explain.semantic_classes == ['MIP','HIP','shower','michel','diffuse']
    assert explain.planes == ['u', 'v', 'y']

def test_make_linear_decoder():
    e = ExplainNetwork(test=True)

    encoder = e.model.encoder
    encoder_size = encoder.net[e.planes[0]][0].net[0].weight.shape[0]
    decoder = e.linear_decoder(in_shape=encoder_size)
    data = e.load.unpack(e.data)[0]
    forward = encoder.forward(data)
    out = decoder.forward(forward)

    assert out is not None 
    assert type(out) == dict 
    assert list(out.keys()) == e.planes
    for plane in e.planes: 
        assert out[plane].shape == data[plane].shape

def test_all_static_decoders(): 
    e = ExplainNetwork(test=True)
    decoder_out = e.step_network()
    assert None not in decoder_out

    for output in decoder_out: 
        for _, plane in enumerate(e.planes): 
            expected_size = decoder_out[-1]['x_semantic'][plane].shape
            print(output)
            assert output[plane].shape == expected_size
        