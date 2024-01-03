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

    def _single_plot(self, plot, history, xtick_labels): 
        index = range(len(history))
        plot.scatter(index, history)
        plot.plot(index, history)
        plot.set_xticks(
                ticks=index, 
                labels=xtick_labels, 
                rotation=45, 
                ha="right",
                rotation_mode="anchor")
        plot.grid(which="major", axis='both')


    def visualize(self, explaination=None, file_name="explaination_graph"): 
        """ 
        Produce a visualization of the explaination
        """

        plt.close("all")
        figure, subplots = plt.subplots(2, len(self.planes), figsize=(len(self.planes)*6, 2*6), sharey='row')

        for index, plane in enumerate(self.planes): 
            loss = [self.loss[i][plane] for i in self.loss.keys() if i not in ['planar_decoded', 'nexus_decoded', 'output_decoded']]
            loss_index = [i for i in self.loss.keys() if i not in ['planar_decoded', 'nexus_decoded', 'output_decoded']]

            for step in range(self.message_passing_steps): 
                loss.append(self.loss["planar_decoded"][step][plane])
                loss_index.append(f"planar_decoded_step{step}")
                loss.append(self.loss["nexus_decoded"][step][plane])
                loss_index.append(f"nexus_decoded_step{step}")

            loss.append(self.loss['output_decoded'][plane])
            loss_index.append("output_decoded")
            
            entropy = [self.entropy[i][plane] for i in self.entropy.keys() if i not in ['planar_to_nexus',  'nexus_to_planar', 'nexus_to_decoder']]
            entropy_index = [i for i in self.entropy.keys() if i not in ['planar_to_nexus',  'nexus_to_planar', 'nexus_to_decoder']]

            for step in range(self.message_passing_steps): 
                entropy.append(self.entropy['planar_to_nexus'][step][plane])
                entropy_index.append(f'planar_to_nexus_step{step}')
                entropy.append(self.entropy['nexus_to_planar'][step][plane])
                entropy_index.append(f'nexus_to_planar_step{step}')

            entropy.append(self.entropy['nexus_to_decoder'][plane])
            entropy_index.append('nexus_to_decoder')

            self._single_plot(subplots[0, index], loss, loss_index)
            self._single_plot(subplots[1, index], entropy, entropy_index)

            subplots[0, index].set_title(plane)

 
        subplots[0, 0].set_ylabel("Loss Difference")
        subplots[1, 0].set_ylabel("Information Gain")

        figure.supxlabel("Inference Steps")
        plt.tight_layout()
        plt.savefig(f"{self.out_path.rstrip('/')}/{file_name}_information_gain.png")
    