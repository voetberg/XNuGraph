from collections.abc import Iterable
import torch
import pytorch_lightning as pl
import h5py

from nugraph.models import NuGraph2
from nugraph.data import H5DataModule
from nugraph import util
from torch_geometric.loader import DataLoader

from torch_geometric.data import Batch, HeteroData
from pynuml.io import H5Interface

class Load: 
    def __init__(self,
                 checkpoint_path="/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt", 
                 data_path="/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023.gnn.h5", 
                 batch_size=1, 
                 test=False) -> None:
        
        if test: 
            checkpoint_path = None 
            data_path = "./test_data.h5"
        self.model = self.load_checkpoint(checkpoint_path) if checkpoint_path is not None else NuGraph2(in_features=6, checkpoint=False)
        self.data = self.load_data(data_path, batch_size)

        self.planes = ['u', 'v', 'y']
        self.predictions = self.make_predictions()

    def load_checkpoint(self, checkpoint_path, graph=NuGraph2): 
        # Assumed pre-trained model that can perform inference on the loaded data
        try: 
            model = graph.load_from_checkpoint(
                checkpoint_path, 
                planar_features=64,
                nexus_features = 16,
                vertex_features= 40) 

        except RuntimeError: 
            model =  graph.load_from_checkpoint(
                checkpoint_path,  
                planar_features=64,
                nexus_features = 16,
                vertex_features= 40, 
                map_location=torch.device('cpu'))
        model.eval() 
        return model 

    def load_data(self, data_path, batch_size=1): 
        return H5DataModule(data_path, batch_size=batch_size).val_dataloader()
    
    def load_test_data(self, data_path, batch_size=1): 
        with h5py.File(data_path, "a") as f: 
            data = list(f['dataset']) 
            if "samples/train" not in f: 
                f['samples/train'] = data
                f['samples/validation'] = data
                f['samples/test'] = data
            if "datasize/train" not in f: 
                f['datasize/train'] = [ 0 for _ in range(len(data)) ]

            f.close()
        try:
            dataset = H5DataModule(data_path, batch_size).val_dataloader()
        except: 
            H5DataModule.generate_norm(data_path, batch_size)
            dataset = H5DataModule(data_path, batch_size).val_dataloader()

        return dataset

    @staticmethod
    def unpack(data_batch, planes=['u', 'v', 'y']): 
        try: 
            data_batch = Batch.from_data_list([datum for datum in data_batch])
        except NotImplementedError: 
            # Isn't an iterable
            pass 
        
        return (data_batch.collect('x'), 
                { p: data_batch[p, 'plane', p].edge_index for p in planes }, 
                { p: data_batch[p, 'nexus', 'sp'].edge_index for p in planes }, 
                torch.empty(data_batch['sp'].num_nodes, 0), 
                { p: data_batch[p].batch for p in planes }
                )

    def make_predictions(self): 
        accelerator, device = util.configure_device()
        trainer = pl.Trainer(accelerator=accelerator,
                            logger=False)
        predictions = trainer.predict(self.model, dataloaders=self.data)
        return predictions
    
    def save_mini_batch(self): 
        batch = self.data.dataset
        h5_file = h5py.File(name="./test_data.h5", mode="w")
        interface = H5Interface(h5_file)
        interface.save("validation", batch)
        
        with h5_file as f: 

            f['planes'] = self.planes
            f["semantic_classes"] = ['MIP','HIP','shower','michel','diffuse']

        f.close()


if __name__ == "__main__": 

    #H5DataModule.generate_samples("./test_data.h5")#, batch_size=16)
    Load(checkpoint_path=None, data_path="test_data.h5", test=True, batch_size=1)
