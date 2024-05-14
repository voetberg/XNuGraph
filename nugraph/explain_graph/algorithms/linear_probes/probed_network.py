import torch
import numpy as np
import os
import matplotlib.pyplot as plt

import json
from tqdm import tqdm

from nugraph.explain_graph.algorithms.linear_probes.linear_decoder import (
    DynamicLinearDecoder,
)
from nugraph.explain_graph.algorithms.linear_probes.latent_representation import (
    LatentRepresentation,
)

from nugraph.util import RecallLoss
from nugraph.explain_graph.algorithms.linear_probes.feature_loss import FeatureLoss


class DynamicProbedNetwork:
    def __init__(
        self,
        model,
        data,
        planes=["u", "v", "y"],
        semantic_classes=["MIP", "HIP", "shower", "michel", "diffuse"],
        loss_metric=RecallLoss(),
        epochs: int = 25,
        message_passing_steps=5,
        out_path="./",
        make_latent_rep=False,
        make_embedding_rep=True,
    ) -> None:
        self.model = model
        self.data = data
        self.planes = planes
        self.semantic_classes = semantic_classes
        self.loss_metric = loss_metric
        self.out_path = out_path
        self.message_passing_steps = message_passing_steps
        self.epochs = epochs
        self.make_latent_rep = make_latent_rep
        self.make_embedding_rep = make_embedding_rep

        self.training_history = {}
        self.make_probes()

        if not os.path.exists(os.path.dirname(self.out_path)):
            os.makedirs(self.out_path)

    def encoder_in_func(self, x):
        x, _, _, _, _ = self.model.unpack_batch(x)
        x = {plane: x[plane][:, : self.model.in_features] for plane in self.planes}
        return self.model.encoder(x)

    def message_in_function(self, x, step):
        m = self.encoder_in_func(x)
        feat, edge_index_plane, edge_index_nexus, nexus, _ = self.model.unpack_batch(x)
        feat = {
            plane: feat[plane][:, : self.model.in_features] for plane in self.planes
        }

        for _ in range(step + 1):
            for _, p in enumerate(self.planes):
                s = feat[p].detach().unsqueeze(1).expand(-1, m[p].size(1), -1)
                m[p] = torch.cat((m[p], s), dim=-1)
            self.model.plane_net(m, edge_index_plane)
            self.model.nexus_net(m, edge_index_nexus, nexus)
        return m

    def decoder_in_func(self, x):
        m = self.message_in_function(x, step=self.message_passing_steps)
        _, _, _, _, batch = self.model.unpack_batch(x)

        decoder_out = self.model.semantic_decoder(m, batch)["x_semantic"]
        return decoder_out

    def make_probes(self):
        probes = {}

        for loss_name in ["tracks", "hipmip"]:
            loss_function = FeatureLoss(feature=loss_name).loss
            probes[loss_name] = {}

            encoder_probe = DynamicLinearDecoder(
                in_shape=(self.model.planar_features, 1),
                input_function=self.encoder_in_func,
                loss_function=loss_function,
            )
            probes[loss_name]["encoder"] = encoder_probe

            for step in range(self.message_passing_steps):

                def internal_message_step(x):
                    return self.message_in_function(x, step)

                message_probe = DynamicLinearDecoder(
                    in_shape=(self.model.planar_features, 1),
                    input_function=internal_message_step,
                    loss_function=loss_function,
                )
                probes[loss_name][f"message_{step+1}"] = message_probe

            decoder_probe = DynamicLinearDecoder(
                in_shape=(len(self.semantic_classes), 1),
                input_function=self.decoder_in_func,
                loss_function=loss_function,
            )
            probes[loss_name]["decoder"] = decoder_probe

        self.probes = probes

    def train(self):
        # Do these first, as they can be done with the original checkpoint and nothing else
        layer_rep_outpath = f"{self.out_path.rstrip('/')}/clustering"
        if not os.path.exists(layer_rep_outpath):
            os.makedirs(layer_rep_outpath)

        if self.make_latent_rep:
            weights = self.extract_network_weights()
            for name, weight in weights.items():
                plot_name = f"weights_{name}"
                title = f"Layer weights - {name.upper()}"
                LatentRepresentation(
                    weight, out_path=layer_rep_outpath, name=plot_name, title=title
                )()

        if self.make_embedding_rep:
            embeddings = self.extract_feature_embedding()
            labels = {
                p: torch.concatenate([batch[p]["y_semantic"] for batch in self.data])
                for p in self.planes
            }

            for name, embedding in embeddings.items():
                plot_name = f"feature_embedding_{name}"
                title = f"Feature Embedding - {name.upper()}"
                LatentRepresentation(
                    embedding,
                    true_labels=labels,
                    out_path=layer_rep_outpath,
                    name=plot_name,
                    title=title,
                )()

        # for loss_function, decoder_step in self.probes.items():
        #     self.training_history[loss_function] = {}
        #     for decoder_name, probe in decoder_step.items():
        #         trainer = TrainSingleProbe(probe=probe, epochs=self.epochs)
        #         loss = trainer.train_probe(self.data)
        #         self.training_history[loss_function][decoder_name] = loss
        #         self.save_progress()

    def extract_network_weights(self):
        weights = {}

        def compress_class_linear(class_linear):
            weights = torch.concat(
                [
                    class_linear.net[i].weight.unsqueeze(dim=-1)
                    for i in range(class_linear.num_classes)
                ],
                axis=-1,
            )
            if weights.shape[0] != 1:
                weights = weights.detach().reshape(
                    weights.shape[0], int(np.prod(weights.shape) / weights.shape[0])
                )
            else:
                weights = torch.swapaxes(
                    torch.swapaxes(weights, 0, -1).squeeze(dim=-1), 0, -1
                ).detach()
            return weights

        weights["encoder"] = {
            p: compress_class_linear(self.model.encoder.net[p][0]) for p in self.planes
        }
        weights["message"] = {
            p: compress_class_linear(self.model.nexus_net.nexus_down[p].node_net[-2])
            for p in self.planes
        }
        weights["decoder"] = {
            p: compress_class_linear(self.model.semantic_decoder.net[p])
            for p in self.planes
        }
        return weights

    def extract_feature_embedding(self):
        embedding = {}

        def extract_embedding(layer_func):
            total_embedding = []
            for batch in self.data:
                embedding = layer_func(batch)

                total_embedding.append(embedding)

            total_embedding = {
                p: torch.concat(
                    [
                        embed[p]
                        .reshape(
                            embed[p].shape[0],
                            int(np.prod(embed[p].shape) / embed[p].shape[0]),
                        )
                        .detach()
                        for embed in total_embedding
                    ]
                )
                for p in self.planes
            }
            print(total_embedding["u"].shape)
            return total_embedding

        embedding["encoder"] = {
            p: embed for p, embed in extract_embedding(self.encoder_in_func).items()
        }
        embedding["message"] = {
            p: embed
            for p, embed in extract_embedding(
                lambda x: self.message_in_function(x, self.message_passing_steps)
            ).items()
        }
        # embedding['decoder'] = {p: embed for p, embed in extract_embedding(self.decoder_in_func).items()}
        return embedding

    def save_progress(self):
        with open(f"{self.out_path}/probe_history.json", "w") as f:
            json.dump(self.training_history, f)

    def plot_probe_training_history(self, file_name=""):
        plt.close("all")

        metrics = self.training_history.keys()
        fig, subplots = plt.subplots(
            ncols=len(metrics),
            nrows=1,
            figsize=((2 * len(metrics)) + 10, 10),
        )

        for subplot, metric in zip(subplots, metrics):
            probe_history = self.training_history[metric]
            for probe_name, history in probe_history.items():
                subplot.plot(history, marker="o", label=probe_name)

            subplot.set_title(metric)

        fig.supxlabel("Training Epoch")
        fig.supylabel("Loss")
        fig.tight_layout()
        plt.legend()
        plt.savefig(f"{self.out_path.rstrip('/')}/{file_name}_probe_loss.png")


class TrainSingleProbe:
    def __init__(
        self,
        probe: DynamicLinearDecoder,
        planes: list = ["v", "u", "y"],
        epochs: int = 25,
    ) -> None:
        self.probe = probe
        self.planes = planes
        self.epochs = epochs
        self.optimizer = torch.optim.SGD(params=self.probe.parameters(), lr=0.001)

    def step(self, x, labels):
        self.probe.train(True)
        prediction = self.probe.forward(x)
        loss = self.probe.loss(prediction, labels)
        for plane in loss:
            plane.backward(retain_graph=True)
        self.optimizer.step()
        return loss

    def train_probe(self, data):
        loss = []
        for _ in tqdm(range(self.epochs)):
            epoch_loss = []
            for batch in data:
                epoch_loss.append(self.step(batch, batch))
            loss.append(torch.mean(torch.tensor(epoch_loss)).item())
        return loss
