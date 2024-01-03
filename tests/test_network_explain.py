import pytest 
import torch 

from nugraph.explain_graph.algorithms.linear_probes.linear_decoder import LinearDecoder
from nugraph.explain_graph.algorithms.linear_probes.probed_network import ProbedNetwork

from nugraph.explain_graph.utils.load import Load 


def test_init_abstract_defaults(): 
    explain = ProbedNetwork(model=Load(test=True).model)
    assert hasattr(explain, 'model')

    assert explain.semantic_classes == ['MIP','HIP','shower','michel','diffuse']
    assert explain.planes == ['u', 'v', 'y']

def test_make_linear_decoder():
    load = Load(test=True)
    e = ProbedNetwork(model=load.model)
    
    encoder = e.model.encoder
    encoder_size = encoder.net[e.planes[0]][0].net[0].weight.shape[0]
    decoder = LinearDecoder(in_shape=encoder_size, planes=e.planes, num_classes=len(e.semantic_classes))
    data = load.unpack(load.data)[0]

    forward = encoder.forward(data)
    print("encoder_out", forward['u'].shape)
    out = decoder.forward(forward)

    assert out is not None 
    assert type(out) == dict 
    assert list(out.keys()) == e.planes
    out_classes = decoder.classes(out)
    data = next(iter(load.data))

    for plane in e.planes: 
        assert out_classes[plane].shape == data[plane]['y_semantic'].shape

def test_all_static_decoders(): 
    load = Load(test=True)
    e = ProbedNetwork(model=load.model)

    input_decoder, encoder_decoder, planar_decoder, nexus_decoder, output = e.step_network(load.data)
    for _, plane in enumerate(e.planes): 
        expected_size = output[plane].shape
        assert input_decoder[plane].shape == expected_size
        
        assert encoder_decoder[plane].shape == expected_size
        assert planar_decoder[0][plane].shape == expected_size
        assert nexus_decoder[0][plane].shape == expected_size

    # Same thing with the softmax 
    input_decoder, encoder_decoder, planar_decoder, nexus_decoder, output = e.step_network(load.data, apply_softmax=True)
    for _, plane in enumerate(e.planes): 
        expected_size = torch.Size([output[plane].shape[0]])
        assert input_decoder[plane].shape == expected_size
        
        assert encoder_decoder[plane].shape == expected_size
        assert planar_decoder[0][plane].shape == expected_size
        assert nexus_decoder[0][plane].shape == expected_size


def test_multiple_steps(): 
    load = Load(test=True)
    e = ProbedNetwork(model=load.model)
    n_steps = 10
    _, _, planar_decoder, nexus_decoder, _ = e.step_network(load.data, message_passing_steps=n_steps)

    assert len(planar_decoder.keys()) == n_steps
    assert len(nexus_decoder.keys()) == n_steps