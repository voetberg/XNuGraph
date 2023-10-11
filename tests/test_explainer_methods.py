import pytest 
import os 
from nugraph.explain_graph.gnn_explain import GNNExplain


from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

data_path = "./tests/resources/test_data.h5"
out_path = "./explainations"
checkpoint_path = "./tests/resources/paper.ckpt"
 
class TestExplain(GNNExplain):
    def __init__(self):
        super().__init__(data_path, out_path, checkpoint_path, batch_size=16)

    def load_checkpoint(self, checkpoint_path: str):
        num_features = self.data.num_features
        num_classes = self.data.num_classes
        class GCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(num_features, 16)
                self.conv2 = GCNConv(16, num_classes)

            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)
                return F.log_softmax(x, dim=1)

        return GCN().to(torch.device('cpu'))

    def load_data(self, data_path: str, batch_size: int):
        dataset = 'Cora'
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Planetoid')
        dataset = Planetoid(path, dataset)
        return dataset


def test_has_explaination(): 
    e = TestExplain() 
    assert hasattr(e, "explainer")

def test_inference_single_point(): 
    e = TestExplain() 
    explaination = e.explain(e.data, node_index=[1])
    #Not sure what I should expect out of these. 
    assert len(explaination) == len(e.data)

def test_inference_multi_point(): 
    e = TestExplain() 
    explaination = e.explain(e.data, node_index=[1,2,3])
    assert len(explaination) == len(e.data) 

def test_visualizations_single_image(): 
    e = TestExplain() 
    explaination = e.explain(e.data)
    file_name = "test_file.png"
    e.visualize(explaination, file_name=file_name)

    assert os.path.exists(f"{out_path}/{file_name}")

def test_visualizations_batch(): 
    e = TestExplain() 
    e.inference()
    e.visualize()

    assert os.listdir(f"{out_path}/plots") == len(e.data)

def test_save_results_csv(): 
    e = TestExplain() 
    e.inference() 
    e.save(file_name="explainations", format='csv')

    assert os.path.exists(f"{e.out_path}/explainations.csv")

def test_save_results_h5(): 
    e = TestExplain() 
    e.inference() 
    e.save(file_name="explainations", format='hdf') 

    assert os.path.exists(f"{e.out_path}/explainations.hdf")
