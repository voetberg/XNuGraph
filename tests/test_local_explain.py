import pytest
import torch 
import os 

from nugraph.explain_graph import ExplainLocal
from nugraph.data import H5DataModule

checkpoint_path = "paper.ckpt" # Not for remote ci/cd
data_path = ""

@pytest.fixture
def Child(): 
    class thing(ExplainLocal): 
        def __init__(self, data_path: str,checkpoint_path: str = None, out_path: str = "explainations/",  batch_size: int = 16) -> None:
            super().__init__(data_path, out_path, checkpoint_path, batch_size)
    return thing 

def test_load_model(Child): 
    model = Child(data_path=data_path, checkpoint_path=checkpoint_path).model 
    assert isinstance(model, torch.nn.Module)

def test_load_data(Child): 
    data = Child(data_path=data_path, checkpoint_path=checkpoint_path).data 
    assert isinstance(data, H5DataModule)

def test_no_model(Child): 
    model = Child(data_path=data_path).model 
    assert isinstance(model, torch.nn.Module)

def test_visualize_subclass(): 
    class Child(ExplainLocal): 
        def __init__(self, checkpoint_path: str = None, out_path: str = "explainations/",  batch_size: int = 16) -> None:
            super().__init__(data_path, out_path, checkpoint_path, batch_size)

        def visualize(self, path): 
            open(path, "w")
    
    path = "test_file.png"
    Child().plot(path)
    if os.path.exists(path): 
        assert True 
        os.remove(path)
    else: 
        assert False


def test_save_results_subclass(): 
    #TODO basic save structure for the cluster. I don't know what it should be 
    # Yet.
    pass 