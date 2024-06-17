import torch
import numpy as np
import os

import json
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from pathlib import Path

from nugraph.explain_graph.algorithms.linear_probes.linear_decoder import (
    DynamicLinearDecoder,
)
from nugraph.explain_graph.algorithms.linear_probes.latent_representation import (
    LatentRepresentation,
)

from nugraph.util import RecallLoss
from nugraph.explain_graph.algorithms.linear_probes.feature_loss import FeatureLoss


from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, barrier


def group_setup(device, total_devices):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8090"
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
        inference=False,
        multicore=True, 
        test=False, 
    ) -> None:
        
        torch.set_float32_matmul_precision('high')

        batch_size = 64 if not test else 2
        if multicore: 
            group_setup(rank, total_devices)
            self.data = DataLoader(data, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(data))
        
        else: 
            self.data = DataLoader(data, batch_size=batch_size, shuffle=True)


        self.device = rank
        self.model = model.to(rank)
        self.model.train(False)
        self.model.freeze()
        self.test = test 

        self.planes = planes
        self.semantic_classes = semantic_classes
        self.loss_metric = loss_metric
        self.out_path = out_path

        self.message_passing_steps = message_passing_steps
        self.epochs = epochs

        self.make_latent_rep = make_latent_rep
        self.make_embedding_rep = make_embedding_rep

        self.feature_loss = feature_loss
        self.network_target = network_target

        self.training_history = {}

        self.probe = self.make_probe(1)
        self.probe_name = f"{self.feature_loss}_{self.network_target}_m{self.message_passing_steps}"

        if not os.path.exists(os.path.dirname(self.out_path)):
            os.makedirs(self.out_path)

        if inference: 
            self.load_probe()
            trainer = TrainSingleProbe(probe=self.probe, device=self.device)
            trainer.inference(self.data, probe_name=self.probe_name, outdir=self.out_path)

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

    def make_probe(self, n_out_features):
        input_size = {
            "encoder": (self.model.planar_features, 1),
            "message": (self.model.planar_features, 1),
            "decoder": (len(self.semantic_classes), 1),
        }
        loss_function = FeatureLoss(feature=self.feature_loss, device=self.device).loss

        probe = DynamicLinearDecoder(
            in_shape=input_size[self.network_target],
            out_shape=n_out_features,
            input_function=self.embedding_function(),
            loss_function=loss_function,
            device=self.device,
        )
        return probe

    def embedding_function(self) -> callable:
        return {
            "encoder": self.encoder_in_func,
            "message": self.message_in_function,
            "decoder": self.decoder_in_func,
        }[self.network_target]


    def load_probe(self): 
        self.probe.load_state_dict(torch.load(f"{self.out_path}/{self.probe_name}_probe_weights.pt"))
        self.probe.eval()
        
    def network_clustering(self):
        layer_rep_outpath = f"{self.out_path.rstrip('/')}/clustering"
        if not os.path.exists(layer_rep_outpath):
            os.makedirs(layer_rep_outpath)

        if self.make_embedding_rep:
            labels = {
                p: torch.concatenate([batch[p]["y_semantic"] for batch in self.data])
                for p in self.planes
            }

            plot_name = f"feature_embedding_{self.probe_name}"
            title = f"Feature Embedding - {self.probe_name.split('_')[1].upper()}_{self.probe_name.split('_')[2].upper()}"

            LatentRepresentation(
                embedding_function=self.embedding_function(),
                data_loader=self.data,
                true_labels=labels,
                out_path=layer_rep_outpath,
                name=plot_name,
                title=title,
            ).visualize()
            
            destroy_process_group()


    def train(self):
        self.network_clustering()

        if not os.path.exists(f"{self.out_path}/{self.probe_name}_probe_history.json"):
            trainer = TrainSingleProbe(probe=self.probe, epochs=self.epochs, device=self.device, test=self.test)
            loss = trainer.train_probe(self.data)
            self.training_history = loss
            self.save_progress()
            trainer.inference(self.data, probe_name=self.probe_name, outdir=self.out_path)
        else: 
            print(f"{self.probe_name} already has results, skipping...")
        
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
            f"{self.out_path}/{self.probe_name}_probe_history.json",
            "w",
        ) as f:
            json.dump(self.training_history, f)
        
        torch.save(
            self.probe.state_dict(),
            f"{self.out_path}/{self.probe_name}_probe_weights.pt"
        )

class TrainSingleProbe:
    def __init__(
        self,
        probe: DynamicLinearDecoder,
        planes: list = ["v", "u", "y"],
        epochs: int = 25,
        device=None, 
        test = False, 
    ) -> None:
        self.probe = probe
        self.planes = planes
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(params=self.probe.parameters(), lr=0.01)
        self.device = device 
        self.test = test 

    def train_step(self, batch):
        m = self.probe.input_function(batch)

        prediction = self.probe.forward(m)
        loss = self.probe.loss(prediction, batch)
        return loss

    def train_probe(self, data):
        training_history = []
        self.probe.train(True)
        for epoch in (pbar := tqdm(range(self.epochs))):
            data.sampler.set_epoch(epoch)
            epoch_loss = 0


            for batch in data:
                loss = self.train_step(batch)
                epoch_loss += loss
            
            epoch_loss.backward()
            self.optimizer.step()
            loss = epoch_loss.item()/len(data)
            training_history.append(loss)
            pbar.set_description(f"Loss: {round(loss, 5)}")

        return training_history


    def inference(self, data, outdir, probe_name): 
        losses = FeatureLoss("tracks").included_features.keys()
        active_loss = {loss: [] for loss in losses}

        for batch in tqdm(data, desc="Computing Feature Loss..."):
            m = self.probe.input_function(batch)
            prediction = self.probe.forward(m)

            for loss in losses: 
                active_loss[loss].append(FeatureLoss(loss, device=self.device).loss(prediction, batch).detach().cpu())

        output = {probe_name: {
            loss_name: np.mean(np.array(loss_value)).astype(float)
            for loss_name, loss_value 
            in active_loss.items()
        }}

        inference = f"{outdir}/inference.json"
        try: 
            Path(inference).touch(exist_ok=False)
        except FileExistsError: 
            pass

        with open(inference, "w+") as f:
            try: 
                existing = json.load(f)
            except json.decoder.JSONDecodeError: 
                existing = {}

            existing[probe_name] = output
            json.dump(existing, f)