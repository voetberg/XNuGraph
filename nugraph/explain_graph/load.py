import torch
import pytorch_lightning as pl

from nugraph.models import NuGraph2
from nugraph.data import H5DataModule
from nugraph import util
from torch_geometric.loader import DataLoader

#from torch_geometric.data import Batch

class Load: 
    def __init__(self,
                 checkpoint_path="/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt", 
                 data_path="/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023.gnn.h5", 
                 batch_size=1, 
                 test=False) -> None:
        self.model = self.load_checkpoint(checkpoint_path) if checkpoint_path is not None else NuGraph2()
        self.data = self.load_data(data_path, batch_size)
        if test: 
            self.data = DataLoader(next(iter(self.data)),
                          batch_size=self.batch_size)
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
    
    def unpack(self, data_batch): 
        return (data_batch.collect('x'), 
                { p: data_batch[p, 'plane', p].edge_index for p in self.planes }, 
                { p: data_batch[p, 'nexus', 'sp'].edge_index for p in self.planes }, 
                torch.empty(data_batch['sp'].num_nodes, 0), 
                { p: data_batch[p].batch for p in self.planes }
                )

    def make_predictions(self): 
        accelerator, device = util.configure_device()
        trainer = pl.Trainer(accelerator=accelerator,
                            logger=False)
        predictions = trainer.predict(self.model, dataloaders=self.data)
        return predictions