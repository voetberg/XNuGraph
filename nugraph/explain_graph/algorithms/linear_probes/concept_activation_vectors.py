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

    def track_loss(self, y_hat, y):
        return FeatureLoss(feature="tracks").loss(y_hat, y.collect("y_semantic"))

    def hipmip_loss(self, y_hat, y):
        return FeatureLoss(feature="hipmip").loss(y_hat, y.collect("y_semantic"))

    def michel_presence_loss(self, y_hat, y):
        pass

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
            encoder_probe, epochs=epochs, overwrite=overwrite, test=test
        )
        return history, class_losses

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
        history, class_losses = self.train(
            message_probe, epochs=epochs, overwrite=overwrite
        )
        return history, class_losses
