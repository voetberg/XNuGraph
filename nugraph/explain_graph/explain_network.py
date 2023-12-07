from nugraph.explain_graph.load import Load
from nugraph.explain_graph.algorithms.linear_probes.probed_network import ""

from nugraph.explain_graph.explain_local import ExplainLocal
import torch 

class ExplainNetwork(ExplainLocal):
    def __init__(self, 
                 data_path: str, 
                 out_path: str = "explainations/", 
                 checkpoint_path: str = None, 
                 batch_size: int = 16, 
                 test: bool = False):
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test)
        self.explainer = ""


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