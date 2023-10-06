import tqdm 
import pandas as pd 
import os 
from nugraph import data, models, util
import lightning.pytorch as pl 
import torch 

class ExplainLocal:
    def __init__(self, data_path:str, out_path:str = "explainations/",checkpoint_path:str=None, batch_size:int=16):
        """
        Abstract class 
        Perform a local explaination method on a single datapoint

        Args:
            data_path (str): _description_
            out_path (str, optional): _description_. Defaults to "explainations/".
            checkpoint_path (str, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 16.
        """
        self.model = self.load_checkpoint(checkpoint_path) if checkpoint_path is not None else models.NuGraph2()
        self.data = self.load_data(data_path, batch_size)
        self.explainations = [] 
        self.out_path = out_path

    def load_checkpoint(self, checkpoint_path:str):
        """Load a saved checkpoint to perform inference

        Returns:
            _type_: _description_
        """

        try: 
            model = models.NuGraph2.load_from_checkpoint(
                checkpoint_path, 
                planar_features=64,
                nexus_features = 16,
                vertex_features= 40) 
            model.eval() 

        except RuntimeError: 
            model =  models.NuGraph2.load_from_checkpoint(
                checkpoint_path,  
                planar_features=64,
                nexus_features = 16,
                vertex_features= 40, 
                map_location=torch.device('cpu'))
            model.eval() 
        return model 

    def inference(self): 
        """_summary_
        """
        accelerator, devices = util.configure_device()
        trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                         logger=False)
        predictions = trainer.predict(self.model, dataloaders=self.data.test_dataloader())
        for _, batch in enumerate(tqdm.tqdm(predictions)):
            for data in batch.to_data_list():
                self.explainations.append(self.explain(self.model, data)) 
        
        self.explainations = pd.concat(self.explainations)


    def load_data(self, data_path, batch_size): 
        """_summary_

        Args:
            data_path (_type_): _description_
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        return data.H5DataModule(data_path, batch_size=batch_size)

    def explain(self, *args, **kwds): 
        """_summary_

        Returns:
            _type_: _description_
        """
        raise NotImplemented
    
    def visualize(self, *args, **kwrds): 
        """_summary_
        """
        raise NotImplemented 
    
    def save(self, format:str='hdf'): 
        """_summary_

        Args:
            format (str, optional): _description_. Defaults to 'hdf'.
        """
        assert format in ['hdf', 'csv'], "format must be 'hdf' or 'csv'"
        if not os.path.exists(self.out_path): 
            os.makedirs(self.out_path)

        save_file = f"{self.out_path}results.{format}"
        {
            "hdf":lambda x:x.to_hdf(save_file, format='table'), 
            'csv': lambda x: x.to_csv(save_file)
        }[format](self.explainations)

    def __call__(self, *args, **kwds):
        self.inference()
        self.save() 
