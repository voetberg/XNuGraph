import tqdm 
import os 
import json 
from nugraph.explain_graph.utils.load import Load

from datetime import datetime 

from torch_geometric.explain import metric as pyg_metrics
from nugraph.explain_graph.utils import metrics 

class ExplainLocal:
    def __init__(
            self, 
            data_path:str, 
            out_path:str = "explainations/",
            checkpoint_path:str=None, 
            batch_size:int=16, 
            test:bool=False, 
            n_batches:int=None, 
            message_passing_steps:int=5):
        """
        Abstract class 
        Perform a local explaination method on a single datapoint

        Args:
            data_path (str): path to h5 file with data, to perform inference on
            out_path (str, optional): Folder to save results to. Defaults to "explainations/".
            checkpoint_path (str, optional): Checkpoint to trained model. If not supplied, creates a new model. Defaults to None.
            batch_size (int, optional): Batch size for the data loader. Defaults to 16.
        """
        self.load = Load(data_path=data_path, checkpoint_path=checkpoint_path, batch_size=batch_size, test=test, n_batches=n_batches, message_passing_steps=message_passing_steps)
        self.data = self.load.data
        self.model = self.load.model
        self.metrics = {}

        self.out_path = out_path.rstrip('/')
        if not os.path.exists(self.out_path): 
            os.makedirs(self.out_path, exist_ok=True)
        self.explainer = None

    def process_graph(self, graph):
        return graph 


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

    def calculate_metrics(self, explainations): 
        fidelity_positive, fidelity_negative = metrics.fidelity(self.explainer, explainations)
        characterization = {plane: 
            pyg_metrics.characterization_score(fidelity_positive[plane], fidelity_negative[plane])
            for plane in self.model.planes
        } 
        unfaithfulness = metrics.unfaithfulness(self.explainer, explainations)

        return {
            "fidelity+": fidelity_positive, 
            "fidelity-":fidelity_negative,
            "character":characterization, 
            "unfaithfulness":unfaithfulness
            }

    def save(self, file_name:str=None): 
        """
        Save the results

        Args:
            file_name (str, optional): Name of file. If not supplied, filename is results_$timestamp. Defaults to None.
        """

        if not os.path.exists(self.out_path): 
            os.makedirs(self.out_path)

        if file_name is None: 
            file_name = datetime.now().timestamp()
        try: 
            self.explainer.algorithm.plot_loss(f"{self.out_path}/exp_loss_{file_name}.png")

        except AttributeError: 
            pass 

        
        json.dump(self.metrics, open(f"{self.out_path}/metrics_{file_name}.json", 'w'))