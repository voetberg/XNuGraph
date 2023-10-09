import tqdm 
import pandas as pd 
import os 
from nugraph import data, models, util
import lightning.pytorch as pl 
import torch 
from datetime import datetime 

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
        self.model = self.load_checkpoint(checkpoint_path) if checkpoint_path is not None else models.NuGraph2()
        #self.data = self.load_data(data_path, batch_size)
        self.explainations = [] 
        self.out_path = out_path

    def load_checkpoint(self, checkpoint_path:str):
        """Load a saved checkpoint to perform inference

        Args:
            checkpoint_path (str): Trained checkpoint

        Returns:
            nugraph.modes.NuGraph2: Model from a loaded the checkpoint
        """

        try: 
            model = models.NuGraph2.load_from_checkpoint(
                checkpoint_path, 
                planar_features=64,
                nexus_features = 16,
                vertex_features= 40) 

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
        """
        Perform predictions and explaination for the loaded data using the model
        """
        accelerator, devices = util.configure_device()
        trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                         logger=False)
        predictions = trainer.predict(self.model, dataloaders=self.data.test_dataloader())
        for _, batch in enumerate(tqdm.tqdm(predictions)):
            for data in batch.to_data_list():
                self.explainations.append(self.explain(self.model, data)) 
        
        self.explainations = pd.concat(self.explainations)


    def load_data(self, data_path:str, batch_size:int): 
        """
        Load h5 dataset

        Args:
            data_path (str): location where data is stored
            batch_size (int): number of samples to load into memory at a single time

        Returns:
            nugraph.data.H5DataModule: Loaded data
        """
        return data.H5DataModule(data_path, batch_size=batch_size)

    def explain(self, *args, **kwds): 
        """
        Impliment the explaination method
        """
        raise NotImplemented
    
    def visualize(self, *args, **kwrds): 
        """ 
        Produce a visualization of the explaination
        """
        raise NotImplemented 
    
    def save(self, file_name:str=None, format:str='hdf'): 
        """
        Save the results to and hdf5 or a csv - saves to outpath/file_name.format

        Args:
            file_name (str, optional): Name of file. If not supplied, filename is results_$timestamp. Defaults to None.
            format (str, optional): Type of file to save, ['hdf', 'csv']. Defaults to 'hdf'.
        """
        assert format in ['hdf', 'csv'], "format must be 'hdf' or 'csv'"
        assert isinstance(self.explainations, pd.DataFrame), "No results found, please run explainations.inference before saving"
       
        if not os.path.exists(self.out_path): 
            os.makedirs(self.out_path)

        if file_name is None: 
            file_name = f"results_{datetime.now().timestamp()}"

        save_file = f"{self.out_path}{file_name}.{format}"
        {
            "hdf":lambda x:x.to_hdf(save_file, 's'), 
            'csv': lambda x: x.to_csv(save_file)
        }[format](self.explainations)

    def __call__(self, *args, **kwds):
        self.inference()
        self.save() 
