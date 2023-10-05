import pytest
from nugraph.explain_graph import ExplainLocal
import pytorch 
import os 


def test_abstract(): 
    with pytest.raises(NotImplementedError): 
        ExplainLocal()

def test_load_model(): 
    class Child(ExplainLocal): 
        def __init__(self) -> None:
            super().__init__()

    model = Child().model 
    assert isinstance(model, pytorch.nn.Module)

def test_load_data(): 
    class Child(ExplainLocal): 
        def __init__(self) -> None:
            super().__init__()

    data = Child().data 
    #TODO - Data format? 
    assert isinstance(data, pyg)

def test_visualize_subclass(): 
    class Child(ExplainLocal): 
        def __init__(self) -> None:
            super().__init__()
        
        def plot(self, path): 
            open(path, "w")
    
    path = "test_file.png"
    Child().plot(path)
    if os.path.exists(path): 
        assert True 
        os.remove(path)
    else: 
        assert False


def test_save_results_subclass(): 
    #TODO basic 
    pass 