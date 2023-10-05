from abc import ABC, abstractmethod
from typing import Any
import tqdm 
import pandas as pd 
import os 

import nugraph as ng
import lightning.pytorch as pl 


class ExplainLocal(ABC): 
    def __init__(self, 
                 data_path:str, 
                 out_path:str = "explainations/",
                 checkpoint_path:str=None, 
                 batch_size:int=16) -> None:
        """
        Abstract class 
        Perform a local explaination method on a single datapoint

        Args:
            data_path (str): _description_
            out_path (str, optional): _description_. Defaults to "explainations/".
            checkpoint_path (str, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 16.
        """
        self.model = self.load_checkpoint(checkpoint_path) if checkpoint_path is not None else NuGraph2()
        self.data = self.load_data(data_path, batch_size)
        self.explainations = [] 
        self.out_path = out_path

    def load_checkpoint(self, checkpoint_path:str):
        """Load a saved checkpoint to perform inference

        Returns:
            _type_: _description_
        """
        model = ng.models.NuGraph.load_from_checkpoint(checkpoint_path) 
        model.eval() 
        return model 

    def inference(self): 
        """_summary_
        """
        accelerator, devices = ng.util.configure_device()
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
        return ng.data.H5DataModule(data_path, batch_size=batch_size)

    @abstractmethod
    def explain(self, *args, **kwds): 
        """_summary_

        Returns:
            _type_: _description_
        """
        return "" 
    
    @abstractmethod
    def visualize(self, *args, **kwrds): 
        """_summary_
        """
        pass 
    
    def save(self, format:str='hdf'): 
        """_summary_

        Args:
            format (str, optional): _description_. Defaults to 'hdf'.
        """
        assert format in ['hdf', 'csv'], "format must be 'hdf' or 'csv'"
        if not os.path.exists(self.out_path): 
            os.makedirs(self.out_path)

        save_file = f"{self.out_path}{}"
        {
            "hdf":lambda x:x.to_hdf(save_file, format='table'), 
            'csv': lambda x: x.to_csv(save_file)
        }[format](self.explainations)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.inference()
        self.save() 
