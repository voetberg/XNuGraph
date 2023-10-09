import pytest
import torch 
import os 

from nugraph.explain_graph.explain_local import ExplainLocal
from nugraph.data import H5DataModule

checkpoint_path = "tests/resources/paper.ckpt" # Not for remote ci/cd
data_path = ""

@pytest.fixture
def generate_data(): 
    pass 

def test_load_model(): 
    model = ExplainLocal(data_path=data_path, checkpoint_path=checkpoint_path).model 
    assert isinstance(model, torch.nn.Module)

def test_load_data(): 
    data = ExplainLocal(data_path=data_path, checkpoint_path=checkpoint_path).data 
    assert isinstance(data, H5DataModule)

def test_no_model(): 
    model = ExplainLocal(data_path=data_path).model 
    assert isinstance(model, torch.nn.Module)

def test_visualize_subclass(): 
    class Child(ExplainLocal): 
        def __init__(self, checkpoint_path: str = None, out_path: str = "explainations/",  batch_size: int = 16) -> None:
            super().__init__(data_path, checkpoint_path=None, out_path="./")

        def visualize(self, path): 
            open(path, "w")
    
    path = "test_file.png"
    Child().visualize(path)
    if os.path.exists(path): 
        assert True 
        os.remove(path)
    else: 
        assert False

    assert hasattr(Child(), "model")


def test_save_results_subclass(): 
    import pandas as pd 
    test_outdir = "./test/explainations/"

    class Child(ExplainLocal): 
        def __init__(self) -> None:
            super().__init__(data_path, checkpoint_path=None, out_path=test_outdir)
            self.explainations = pd.DataFrame({1:[0]})

    Child().save("file_name") 
    
    expected_path = f"{test_outdir}file_name.hdf"
    if os.path.exists(expected_path): 
        assert True 
        os.remove(expected_path)
    else: 
        assert False

    Child().save("file_name", format='csv') 
    expected_path = f"{test_outdir}file_name.csv"
    if os.path.exists(expected_path): 
        assert True 
        os.remove(expected_path)
    else: 
        assert False

    Child().save(format='csv') 
    expected_path = os.listdir(test_outdir)
    if len(expected_path)==1:
        assert True 
        os.remove(f"{test_outdir}{expected_path[0]}")
    else: 
        assert False

    import shutil
    shutil.rmtree(test_outdir, ignore_errors=True)