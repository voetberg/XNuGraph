import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import unbatch

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
        feature_loss=["tracks", "hipmip"], 
        network_target=['encoder', 'message', 'decoder']
    ) -> None:
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.train(False)
        self.data = data

        self.planes = planes
        self.semantic_classes = semantic_classes
        self.loss_metric = loss_metric
        self.out_path = out_path

        self.message_passing_steps = message_passing_steps
        self.epochs = epochs

        self.make_latent_rep = make_latent_rep
        self.make_embedding_rep = make_embedding_rep

        self.feature_loss = feature_loss
        self.network_traget = network_target

        self.training_history = {}
        self.probe = self.make_probe()

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

        for _ in range(step):
            # shortcut connect features
            for p in self.planes:
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

    def make_probe(self):
        input_function = {
            'encoder': self.encoder_in_func, 
             'message': lambda x: self.message_in_function(x, self.message_passing_steps), 
             'decoder': self.decoder_in_func
        }
        input_size = {
            'encoder': (self.model.planar_features, 1), 
             'message': (self.model.planar_features, 1), 
             'decoder': (len(self.semantic_classes), 1)
        }
        loss_function = FeatureLoss(feature=self.feature_loss).loss

        probe = DynamicLinearDecoder(
            in_shape=input_size[self.network_traget],
            input_function=input_function[self.network_traget],
            loss_function=loss_function,
            device=self.device
            )
        return probe

    def network_clustering(self): 
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
                ).visualize()

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
                ).visualize()

    def train(self):
        self.network_clustering()
        trainer = TrainSingleProbe(probe=self.probe, epochs=self.epochs)
        loss = trainer.train_probe(self.data)
        self.training_history = loss
        self.save_progress()

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

            n_batches_use = 20
            for n_batch, batch in enumerate(self.data):
                if n_batch < n_batches_use: 
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
        return embedding

    def save_progress(self):
        with open(f"{self.out_path}/{self.feature_loss}_{self.network_traget}_probe_history.json", "w") as f:
            json.dump(self.training_history, f)

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

    def train_step(self, batch):
        self.probe.train(True)
        prediction = self.probe.forward(batch)
        loss = self.probe.loss(prediction, batch)
        return loss

    def train_probe(self, data):
        training_history = []
        for _ in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(data):
                loss = self.train_step(batch)
                epoch_loss += loss
            epoch_loss.backward()
            self.optimizer.step()
            training_history.append(epoch_loss.item())

        return training_history