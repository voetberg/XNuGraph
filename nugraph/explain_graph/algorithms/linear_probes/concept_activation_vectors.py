"""
Evaluate how well a network can learn a specific feature in a location, and then visualize it with activation maximization
"""

from nugraph.explain_graph.algorithms.linear_probes.feature_loss import FeatureLoss
from nugraph.explain_graph.algorithms.linear_probes.probed_network import (
    DynamicProbedNetwork,
)


class ConceptActivateVectors(DynamicProbedNetwork):
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
        loss_function="tracks",
        include_other_losses=True,
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
        self.included_losses = {
            "tracks": self.track_loss,
            "hipmip": self.hipmip_loss,
            "michel_conservation": self.michel_presence_loss,
            "michel_energy": self.michel_energy_loss,
        }

        try:
            self.loss_function = self.included_losses[loss_function]
        except KeyError:
            print(
                f"{loss_function} not included as a loss function, choose from {self.included_losses.keys()}"
            )
        self.feature_loss = FeatureLoss(feature=loss_function, device=self.device)
        self.include_metrics = (
            [func for key, func in self.included_losses.items() if key != loss_function]
            if include_other_losses
            else []
        )

    def track_loss(self, y_hat, y):
        return self.feature_loss.loss(
            y_hat, y.collect("y_semantic"), loss_func="tracks"
        )

    def hipmip_loss(self, y_hat, y):
        return self.feature_loss.loss(
            y_hat, y.collect("y_semantic"), loss_func="hipmip"
        )

    def michel_presence_loss(self, y_hat, y):
        return self.feature_loss.loss(
            y_hat, y.collect("y_semantic"), loss_func="michel_conservation"
        )

    def michel_energy_loss(self, y_hat, y):
        return self.feature_loss.loss(y_hat, y, loss_func="michel_energy")

    def train_encoder(self, epochs, overwrite, test=False):
        history, class_losses = self.train(
            embedding_function=self.encoder_in_func,
            epochs=epochs,
            overwrite=overwrite,
        )
        return history, class_losses

    def train_message(self, message_step, epochs, overwrite):
        history, class_losses = self.train(
            embedding_function=lambda x: self.message_in_function(x, message_step),
            epochs=epochs,
            overwrite=overwrite,
        )
        return history, class_losses

    def train(self, embedding_function, overwrite: bool = False, epochs: int = 25):
        probe = self.make_probe(
            input_features=(self.model.planar_features,),
            embedding_function=embedding_function,
            n_out_features=len(self.semantic_classes),
            loss_function=self.loss_function,
            extra_metrics=self.include_metrics,
        )
        history, extra_losses = super().train(probe, overwrite, epochs)
        return history, extra_losses
