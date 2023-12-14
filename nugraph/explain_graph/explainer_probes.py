from nugraph.explain_graph.utils.load import Load
from nugraph.explain_graph.algorithms.linear_probes.probed_network import ProbedNetwork

from nugraph.explain_graph.explain import ExplainLocal
import torch 
import matplotlib.pyplot as plt 

class ExplainNetwork(ExplainLocal):
    def __init__(self, 
                 data_path: str, 
                 out_path: str = "explainations/", 
                 checkpoint_path: str = None, 
                 planes = ['u', 'v', 'y'],
                 message_passing_steps=5, 
                 batch_size: int = 16, 
                 test: bool = False):
        super().__init__(data_path, out_path, checkpoint_path, batch_size, test)
        self.planes = planes
        self.message_passing_steps = message_passing_steps
        self.explainer = ProbedNetwork(model=self.model, planes=self.planes)

        self.entropy = {}
        self.loss = {}


    def explain(self, data, **kwargs): 
        """
        Impliment the explaination method
        """
        self.entropy, self.loss = self.explainer.forward(data, self.message_passing_steps)

    def _single_plot(self, plot, history): 
        index = range(len(history))
        plot.scatter(index, history)
        plot.plot(index, history)

    def visualize(self, explaination=None, file_name="explaination_graph"): 
        """ 
        Produce a visualization of the explaination
        """

        plt.close("all")
        figure, subplots = plt.subplots(2, len(self.planes), figsize=(len(self.planes)*6, 2*6), sharey='row')

        for index, plane in enumerate(self.planes): 
            loss = [self.loss[i][plane] for i in self.loss.keys() if i not in ['planar_decoded', 'nexus_decoded']]
            for step in range(self.message_passing_steps): 
                loss.append(self.loss["planar_decoded"][step][plane])
                loss.append(self.loss["nexus_decoded"][step][plane])
            
            entropy = [self.entropy[i][plane] for i in self.entropy.keys() if i not in ['planar_to_nexus', 'nexus_to_decoder']]
            entropy +=  [self.entropy['planar_to_nexus'][i][plane] for i in self.entropy['planar_to_nexus']]
            entropy.append(self.entropy['nexus_to_decoder'][plane])
            
            self._single_plot(subplots[0, index], loss)
            self._single_plot(subplots[1, index], entropy)

            subplots[0, index].set_title(plane)

 
        subplots[0, 0].set_ylabel("Loss History")
        subplots[1, 0].set_ylabel("Information Gain")

        figure.supxlabel("Inference Steps")
        plt.savefig(f"{self.out_path.rstrip('/')}/{file_name}_information_gain.png")
    