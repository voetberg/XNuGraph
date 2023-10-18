import tqdm 
import os 
from nugraph import data, models
import h5py

import torch 
from datetime import datetime 

from torch_geometric.explain import Explanation, metric
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import unbatch

class ExplainLocal:
    def __init__(self, data_path:str, out_path:str = "explainations/",checkpoint_path:str=None, batch_size:int=16):
        """
        Abstract class 
        Perform a local explaination method on a single datapoint

        Args:
            data_path (str): path to h5 file with data, to perform inference on
            out_path (str, optional): Folder to save results to. Defaults to "explainations/".
            checkpoint_path (str, optional): Checkpoint to trained model. If not supplied, creates a new model. Defaults to None.
            batch_size (int, optional): Batch size for the data loader. Defaults to 16.
        """
        self.data = self.load_data(data_path, batch_size)

        self.model = self.load_checkpoint(checkpoint_path) if checkpoint_path is not None else models.NuGraph2()
        self.explainations = Explanation()
        self.out_path = out_path.rstrip('/')

        self.explainer = None

    def load_checkpoint(self, checkpoint_path:str):
        """Load a saved checkpoint to perform inference

        Args:
            checkpoint_path (str): Trained checkpoint

        Returns:
            nugraph.modes.NuGraph2: Model from a loaded the checkpoint
        """
        class Graph(models.NuGraph2): 
            def __init__(self, in_features: int = 4, planar_features: int = 8, nexus_features: int = 8, vertex_features: int = 32, planes: list[str] = ..., semantic_classes: list[str] = ..., event_classes: list[str] = ..., num_iters: int = 5, event_head: bool = True, semantic_head: bool = True, filter_head: bool = False, vertex_head: bool = False, checkpoint: bool = False, lr: float = 0.001):
                super().__init__(in_features, planar_features, nexus_features, vertex_features, planes, semantic_classes, event_classes, num_iters, event_head, semantic_head, filter_head, vertex_head, checkpoint, lr)
            
            def forward(self, x: dict, edge_index_plane: dict, edge_index_nexus: dict, nexus, batch: dict):
                label = super().forward(x, edge_index_plane, edge_index_nexus, nexus, batch)

                # Output of the network: 
                # x_semantic: {
                #     u: Tensor,
                #     v: Tensor,
                #     y: Tensor,
                # },
            
                return torch.Tensor([label['x_semantic']['u'], label['x_semantic']['v'],label['x_semantic']['y']])
        
        try: 
            model = Graph.load_from_checkpoint(
                checkpoint_path, 
                planar_features=64,
                nexus_features = 16,
                vertex_features= 40) 

        except RuntimeError: 
            model =  Graph.load_from_checkpoint(
                checkpoint_path,  
                planar_features=64,
                nexus_features = 16,
                vertex_features= 40, 
                map_location=torch.device('cpu'))
        model.eval() 
        return model 

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

    def load_data(self, data_path:str, batch_size:int): 
        """
        Load h5 dataset

        Args:
            data_path (str): location where data is stored
            batch_size (int): number of samples to load into memory at a single time

        Returns:
            nugraph.data.H5Dataset: Loaded data
        """
        return data.H5DataModule(data_path, batch_size=batch_size).val_dataloader()

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
