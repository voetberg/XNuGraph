"""
Evaluate the ability for a network section to learn a the target of the network
"""

from nugraph.explain_graph.algorithms.linear_probes.probed_network import (
    ProbedNetwork,
)

import numpy as np
import matplotlib.pyplot as plt
import torch


class AccuracyActivationVectors(ProbedNetwork):
    def __init__(
        self,
        model,
        data,
        rank,
        total_devices,
        planes=["u", "v", "y"],
        semantic_classes=["MIP", "HIP", "shower", "michel", "diffuse"],
        probe_name=None,
        out_path="./",
        multicore=True,
        test=False,
        batch_size: int = 64,
    ) -> None:
        super().__init__(
            model,
            data,
            rank,
            total_devices,
            planes,
            semantic_classes,
            probe_name,
            out_path,
            multicore,
            test,
            batch_size,
        )
        self.semantic_classes = semantic_classes

    def accuracy_loss(self, y_hat, y):
        loss = 0
        labels = y.collect("y_semantic")
        for p in self.planes:
            y_hat_plane = y_hat[p].type(torch.float)
            y = labels[p].type(torch.long)
            loss += torch.nn.functional.cross_entropy(y_hat_plane, y, ignore_index=-1)

        return loss / len(self.planes)

    def ind_class_loss(self, y_hat, y, class_index):
        loss = 0
        labels = y.collect("y_semantic")
        for p in self.planes:
            true_class = torch.isin(
                labels[p], torch.tensor([class_index], device=self.device)
            ).type(torch.float)
            prediction_class = torch.isin(
                torch.argmax(y_hat[p], axis=1),
                torch.tensor([class_index], device=self.device),
            ).type(torch.float)

            loss += torch.nn.functional.binary_cross_entropy(
                prediction_class, true_class
            )
        return loss.item() / len(self.planes)

    def train_encoder(self, epochs, overwrite, test=False):
        encoder_probe = self.make_probe(
            input_features=(self.model.planar_features,),
            embedding_function=self.encoder_in_func,
            n_out_features=len(self.semantic_classes),
            loss_function=self.accuracy_loss,
            extra_metrics=[
                lambda x, y: self.ind_class_loss(x, y, i)
                for i in range(len(self.semantic_classes))
            ],
        )
        history, class_losses = self.train(
            encoder_probe, epochs=epochs, overwrite=overwrite
        )
        return history, class_losses

    def train_message(self, message_step, epochs, overwrite):
        message_probe = self.make_probe(
            input_features=(self.model.planar_features,),
            embedding_function=lambda x: self.message_in_function(x, message_step),
            n_out_features=len(self.semantic_classes),
            loss_function=self.accuracy_loss,
            extra_metrics=[
                lambda x, y: self.ind_class_loss(x, y, i)
                for i in range(len(self.semantic_classes))
            ],
        )
        history, class_losses = self.train(
            message_probe, epochs=epochs, overwrite=overwrite
        )
        return history, class_losses

    def visualize(self, loss, class_loss, save=False, multiprobe=False):
        """
        Plot either the history for a single probe, or all of them stacked
        """
        if multiprobe:
            if len(np.array(loss).shape) != 2:
                raise ValueError(
                    "Please pass multiple values to visualize multiple probes"
                )

        if save:
            ""
        else:
            plt.show()

    def cluster_latent_space(self):
        pass

    def produce_maximization(self):
        pass
