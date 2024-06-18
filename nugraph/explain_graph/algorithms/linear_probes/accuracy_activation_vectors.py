"""
Evaluate the ability for a network section to learn a the target of the network
"""

from nugraph.explain_graph.algorithms.linear_probes.probed_network import (
    DynamicProbedNetwork,
)
from nugraph.util.RecallLoss import RecallLoss

import numpy as np
import matplotlib.pyplot as plt
import torch


class AccuracyActivationVectors(DynamicProbedNetwork):
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
        loss_func = RecallLoss(ignore_index=-1, num_classes=len(self.semantic_classes))
        labels = y.collect("y_semantic")
        prediction = y_hat.collect("x_semantic")
        for p in self.planes:
            loss += loss_func(np.argmax(prediction[p], axis=1), labels[p])

        return loss / len(self.planes)

    def ind_class_loss(self, y_hat, y, class_index):
        labels = y.collect("y_semantic")
        prediction = y_hat.collect("x_semantic")
        loss = 0
        for p in self.planes:
            true_class = torch.tensor(labels[p] == class_index).to(int)
            prediction_class = torch.tensor(
                np.argmax(prediction[p], axis=1) == class_index
            ).to(int)
            loss += torch.nn.functional.cross_entropy(prediction_class, true_class)
        return loss / len(self.planes)

    def train_encoder(self, epochs, overwrite):
        encoder_probe = self.make_probe(
            input_features=(self.model.planar_features),
            embedding_function=self.encoder_in_func,
            n_out_features=len(self.semantic_classes),
            loss_function=self.accuracy_loss,
            extra_metrics=[
                lambda x, y: self.ind_class_loss(x, y, i)
                for i in range(len(self.semantic_classes))
            ],
        )
        history, inference = self.train(
            encoder_probe, epochs=epochs, overwrite=overwrite
        )
        class_losses = self.ind_class_loss(encoder_probe)
        return history, inference, class_losses

    def train_message(self, message_step, epochs, overwrite):
        message_probe = self.make_probe(
            input_features=self.model.planar_features,
            embedding_function=lambda x: self.message_in_function(x, message_step),
            n_out_features=len(self.semantic_classes),
            loss_function=self.accuracy_loss,
            extra_metrics=[
                lambda x, y: self.ind_class_loss(x, y, i)
                for i in range(len(self.semantic_classes))
            ],
        )
        history, inference = self.train(
            message_probe, epochs=epochs, overwrite=overwrite
        )
        class_losses = self.ind_class_loss(message_probe)
        return history, inference, class_losses

    def visualize(
        self, loss, inference_results, class_loss, save=False, multiprobe=False
    ):
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
