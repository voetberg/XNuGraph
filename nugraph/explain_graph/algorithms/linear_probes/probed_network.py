import torch
import numpy as np
import os

import json
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from nugraph.explain_graph.algorithms.linear_probes.linear_decoder import (
    DynamicLinearDecoder,
)
from nugraph.explain_graph.algorithms.linear_probes.latent_representation import (
    LatentRepresentation,
)

from nugraph.util import RecallLoss
from nugraph.explain_graph.algorithms.linear_probes.feature_loss import FeatureLoss


from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


def group_setup(device, total_devices):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(device)
    init_process_group(backend="nccl", rank=device, world_size=total_devices)

class DynamicProbedNetwork:
    def __init__(
        self,
        model,
        data,
        rank, 
        total_devices,
        planes=["u", "v", "y"],
        semantic_classes=["MIP", "HIP", "shower", "michel", "diffuse"],
        loss_metric=RecallLoss(),
        epochs: int = 25,
        message_passing_steps=5,
        out_path="./",
        make_latent_rep=False,
        make_embedding_rep=True,
        feature_loss=["tracks", "hipmip"],
        network_target=["encoder", "message", "decoder"],
    ) -> None:
        
        group_setup(rank, total_devices)
        self.device = rank
        self.model = model.to(rank)
        self.model.train(False)
        self.model.freeze()
        self.data = DataLoader(data, batch_size=64, shuffle=False, sampler=DistributedSampler(data))

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

    def message_in_function(self, batch):

        x, edge_index_plane, edge_index_nexus, nexus, _ = self.model.unpack_batch(batch)
        x = {plane: x[plane][:, : self.model.in_features] for plane in self.planes}
        m = self.model.encoder(x)

        for _ in range(self.message_passing_steps):
            # shortcut connect features
            for p in self.planes:
                s = x[p].detach().unsqueeze(1).expand(-1, m[p].size(1), -1)
                m[p] = torch.cat((m[p], s), dim=-1)

            self.model.plane_net(m, edge_index_plane)
            self.model.nexus_net(m, edge_index_nexus, nexus)

        return m

    def decoder_in_func(self, x):
        m = self.message_in_function(x)
        _, _, _, _, batch = self.model.unpack_batch(x)

        decoder_out = self.model.semantic_decoder(m, batch)["x_semantic"]
        return decoder_out

    def make_probe(self):
        input_function = {
            "encoder": self.encoder_in_func,
            "message": self.message_in_function,
            "decoder": self.decoder_in_func,
        }
        input_size = {
            "encoder": (self.model.planar_features, 1),
            "message": (self.model.planar_features, 1),
            "decoder": (len(self.semantic_classes), 1),
        }
        loss_function = FeatureLoss(feature=self.feature_loss, device=self.device).loss

        probe = DynamicLinearDecoder(
            in_shape=input_size[self.network_traget],
            input_function=input_function[self.network_traget],
            loss_function=loss_function,
            device=self.device,
        )
        return probe

    def network_clustering(self):
        layer_rep_outpath = f"{self.out_path.rstrip('/')}/clustering"
        if not os.path.exists(layer_rep_outpath):
            os.makedirs(layer_rep_outpath)

        if self.make_embedding_rep:
            labels = {
                p: torch.concatenate([batch[p]["y_semantic"] for batch in self.data])
                for p in self.planes
            }

            embedding_functions = {
                "encode": self.encoder_in_func,
                "message": lambda x: self.message_in_function(
                    x, self.message_passing_steps
                ),
            }
            for name, embedding in embedding_functions.items():
                plot_name = f"feature_embedding_{name}"
                title = f"Feature Embedding - {name.upper()}"

                LatentRepresentation(
                    embedding_function=embedding,
                    data_loader=self.data,
                    true_labels=labels,
                    out_path=layer_rep_outpath,
                    name=plot_name,
                    title=title,
                ).visualize()
        
    def train(self):
        self.network_clustering()
        trainer = TrainSingleProbe(probe=self.probe, epochs=self.epochs, device=self.device)
        loss = trainer.train_probe(self.data)
        self.training_history = loss
        self.save_progress()
        destroy_process_group()

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

    def save_progress(self):
        with open(
            f"{self.out_path}/{self.feature_loss}_{self.network_traget}_probe_history.json",
            "w",
        ) as f:
            json.dump(self.training_history, f)


class TrainSingleProbe:
    def __init__(
        self,
        probe: DynamicLinearDecoder,
        planes: list = ["v", "u", "y"],
        epochs: int = 25,
        device=None
    ) -> None:
        self.probe = probe
        self.planes = planes
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(params=self.probe.parameters(), lr=0.01)
        self.device = device 
        
    def train_step(self, batch):
        m = self.probe.input_function(batch)

        prediction = self.probe.forward(m)
        loss = self.probe.loss(prediction, batch)
        return loss

    def train_probe(self, data):
        training_history = []
        self.probe.train(True)
        for epoch in range(self.epochs):
            data.sampler.set_epoch(epoch)
            epoch_loss = 0

            for batch in  (pbar := tqdm(data)):
                loss = self.train_step(batch)
                epoch_loss += loss
                pbar.set_description(f"Loss: {round(loss.item(), 5)}")

            epoch_loss.backward()
            self.optimizer.step()
            training_history.append(epoch_loss.item()/len(data))

        return training_history
