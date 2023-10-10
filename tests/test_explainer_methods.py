import pytest 
import os 
from nugraph.explain_graph.gnn_explain import GNNExplain

data_path = "./tests/resources/test_data.h5"
out_path = "./explainations"
checkpoint_path = "./tests/resources/paper.ckpt"

def test_has_explaination(): 
    e = GNNExplain(data_path, out_path, checkpoint_path) 
    assert hasattr(e, "explainer")

def test_inference_single_point(): 
    e = GNNExplain(data_path, out_path, checkpoint_path) 
    explaination = e.explain(e.data, mode="single_point", point_index=[1], event_index=[1])
    #Not sure what I should expect out of these. 
    assert len(explaination) == 1 

def test_inference_multi_point(): 
    e = GNNExplain(data_path, out_path, checkpoint_path) 
    explaination = e.explain(e.data, mode="single_point", point_index=[1,2,3],event_index=[1])
    assert len(explaination) == 1 

def test_visualizations_single_image(): 
    e = GNNExplain(data_path, out_path, checkpoint_path) 
    explaination = e.explain(e.data, mode="graph", event_index=[1])
    file_name = "test_file.png"
    e.visualize(explaination, file_name=file_name)

    assert os.path.exists(f"{out_path}/{file_name}")

def test_visualizations_batch(): 
    e = GNNExplain(data_path, out_path, checkpoint_path) 
    e.inference(explaintion_kwargs={"mode":"graph"})
    e.visualize()

    assert os.listdir(f"{out_path}/plots") == len(e.data)

def test_save_results_csv(): 
    e = GNNExplain(data_path, out_path, checkpoint_path) 
    e.inference(explaintion_kwargs={"mode":"graph"}) 
    e.save(file_name="explainations", format='csv')

def test_save_results_h5(): 
    e = GNNExplain(data_path, out_path, checkpoint_path) 
    e.inference(explaintion_kwargs={"mode":"graph"}) 
    e.save(file_name="explainations", format='hdf') 
