import tqdm 
import os 
from nugraph import data, models
from nugraph.explain_graph.load import Load
import h5py

import torch 
from datetime import datetime 

from torch_geometric.explain import Explanation, metric
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import unbatch
from functools import partial

class ExplainLocal:
    def __init__(self, data_path:str, out_path:str = "explainations/",checkpoint_path:str=None, batch_size:int=16, test:bool=False):
        """
        Abstract class 
        Perform a local explaination method on a single datapoint

        Args:
            data_path (str): path to h5 file with data, to perform inference on
            out_path (str, optional): Folder to save results to. Defaults to "explainations/".
            checkpoint_path (str, optional): Checkpoint to trained model. If not supplied, creates a new model. Defaults to None.
            batch_size (int, optional): Batch size for the data loader. Defaults to 16.
        """
        self.load = Load(data_path=data_path, checkpoint_path=checkpoint_path, batch_size=batch_size, test=test)
        self.data = self.load.data
        self.model = self.load.model

        self.explainations = Explanation()
        self.out_path = out_path.rstrip('/')
        if not os.path.exists(self.out_path): 
            os.makedirs(self.out_path)
        self.explainer = None

    def process_graph(self, graph):
        """
        Remove double connections from point to point - the graph is bidirectional and should be treated as such
        """ 
        # def remove_double_connections(edge): 
        #     print(edge)
        #     edge_tuples = [{node1, node2} for node1, node2 in zip(edge[0], edge[1])] 
        #     edge_tuple_counts = {edge_tuples.index(key) for key in edge_tuples}
        #     print(edge_tuples)
        #     print(len(edge[0]))
        #     print(len(edge_tuple_counts))
        #     return edge 
            
        # for plane in self.model.planes: 
        #     plane_edges = graph[plane, 'plane', plane]['edge_index']
        #     nexus_edges = graph[plane, 'nexus', 'sp']['edge_index']
        #     graph[plane, 'plane', plane]['edge_index'] = remove_double_connections(plane_edges)
        #     graph[plane, 'nexus', 'sp']['edge_index'] = remove_double_connections(nexus_edges)

        return graph 

    def unpack(self, data): 
        """Unpack the data to be used by the model"""
        return (
            data.collect('x'),
            { p: data[p, 'plane', p].edge_index for p in self.model.planes }, 
            { p: data[p, 'nexus', 'sp'].edge_index for p in self.model.planes },
            torch.empty(data['sp'].num_nodes, 0),
            { p: data[p].batch for p in self.model.planes }
            )


    def inference(self, explaintion_kwargs=None): 
        """
        Perform predictions and explaination for the loaded data using the model
        """
        explaintion_kwargs = {} if explaintion_kwargs is None else explaintion_kwargs

        for _, batch in enumerate(tqdm.tqdm(self.data)):
            explaination = self.explain(batch, raw=False, **explaintion_kwargs)
            self.explainations.update(explaination)

    def explain(self, data, **kwargs): 
        """
        Impliment the explaination method
        """
        raise NotImplemented
    
    def visualize(self, *args, **kwrds): 
        """ 
        Produce a visualization of the explaination
        """
        raise NotImplemented 
    
    def metrics(self, explainations): 
        fidelity_positive, fidelity_negative = metric.fidelity(self.explainer, explainations)
        characterization = metric.characterization_score(fidelity_positive, fidelity_negative)
        unfaithfulness = metric.unfaithfulness(self.explainer, explainations)

        return {
            "fidelity+": fidelity_positive, 
            "fidelity-":fidelity_negative,
            "character":characterization, 
            "unfaithfulness":unfaithfulness
            }

    def save(self, file_name:str=None): 
        """
        Save the results to hdf5 - saves to outpath/file_name.h5

        Args:
            file_name (str, optional): Name of file. If not supplied, filename is results_$timestamp. Defaults to None.
        """
        assert len(self.explainations)!=0, "No results found, please run explainations.inference before saving"

        if not os.path.exists(self.out_path): 
            os.makedirs(self.out_path)

        if file_name is None: 
            file_name = f"results_{datetime.now().timestamp()}"

        save_file = f"{self.out_path}/{file_name}.h5"
        save_results = h5py.File(save_file, 'w')
        for header, data in self.explainations.to_dict().items():
            save_results.create_dataset(header, data=data)

        save_results.close()

    def __call__(self, *args, **kwds):
        self.inference()
        self.save() 
