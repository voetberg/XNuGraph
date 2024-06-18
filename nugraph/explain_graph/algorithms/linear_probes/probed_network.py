from typing import Optional, Sequence
import torch
import numpy as np
import os

import json
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from nugraph.explain_graph.algorithms.linear_probes.linear_decoder import (
    DynamicLinearDecoder,
)

from nugraph.explain_graph.algorithms.linear_probes.feature_loss import FeatureLoss


from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


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
        probe_name=None,
        out_path="./",
        multicore=True,
        test=False,
        batch_size: int = 64,
    ) -> None:
        torch.set_float32_matmul_precision("high")
        batch_size = batch_size if not test else 2

        if multicore:
            group_setup(rank, total_devices)
            self.data = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=False,
                sampler=DistributedSampler(data),
            )

        else:
            self.data = DataLoader(data, shuffle=True)

        self.device = rank
        self.model = model.to(rank)
        self.model.train(False)
        self.model.freeze()
        self.test = test

        self.planes = planes
        self.out_path = out_path

        self.training_history = {}
        self.probe_name = probe_name

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

    def make_probe(
        self,
        input_features,
        embedding_function,
        n_out_features,
        loss_function,
        extra_metrics,
    ):
        probe = DynamicLinearDecoder(
            in_shape=input_features,
            out_shape=n_out_features,
            input_function=embedding_function,
            loss_function=loss_function,
            extra_metrics=extra_metrics,
            device=self.device,
        )
        return probe

    def load_probe(self):
        self.probe.load_state_dict(
            torch.load(f"{self.out_path}/{self.probe_name}_probe_weights.pt")
        )
        self.probe.eval()

    def destroy_gpu_group(self):
        try:
            destroy_process_group()  # Not doing multicore, there is no pg to destroy
        except AssertionError:
            pass

    def train(
        self,
        probe: type[DynamicLinearDecoder],
        overwrite: bool = False,
        epochs: int = 25,
    ):
        if (
            os.path.exists(f"{self.out_path}/{self.probe_name}_probe_history.json")
            and not overwrite
        ):
            print(f"{self.probe_name} already has results, skipping...")
            self.destroy_gpu_group()
        else:
            trainer = TrainSingleProbe(
                probe=probe, epochs=epochs, device=self.device, test=self.test
            )
            loss, metrics = trainer.train_probe(self.data)
            inference = trainer.inference(self.data)
            self.save_progress(loss, inference)
            self.destroy_gpu_group
            return loss, metrics, inference

    def save_progress(self, training_history, inference):
        with open(
            f"{self.out_path}/{self.probe_name}_probe_history.json",
            "w",
        ) as f:
            json.dump(training_history, f)

        with open(f"{self.out_path}/inference.json", "w+") as f:
            try:
                existing = json.load(f)
            except json.decoder.JSONDecodeError:
                existing = {}

            existing[self.probe_name] = inference
            json.dump(existing, f)

        torch.save(
            self.probe.state_dict(),
            f"{self.out_path}/{self.probe_name}_probe_weights.pt",
        )


class TrainSingleProbe:
    def __init__(
        self,
        probe: DynamicLinearDecoder,
        planes: list = ["v", "u", "y"],
        epochs: int = 25,
        device=None,
        test=False,
    ) -> None:
        self.probe = probe
        self.planes = planes
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(params=self.probe.parameters(), lr=0.01)
        self.device = device
        self.test = test

    def train_step(self, batch) -> tuple[float, Optional[Sequence[float]]]:
        m = self.probe.input_function(batch)
        prediction = self.probe.forward(m)
        loss = self.probe.loss(prediction, batch)
        metrics = None
        if hasattr(self.probe, "metrics"):
            metrics = []
            for metric in self.probe.metrics:
                metrics.append(metric(prediction, batch))
        return loss, metrics

    def train_probe(self, data):
        training_history = []
        metric_history = []

        self.probe.train(True)
        for epoch in (pbar := tqdm(range(self.epochs))):
            if hasattr(data.sampler, "set_epoch"):
                data.sampler.set_epoch(epoch)

            epoch_loss = 0
            metrics = []
            for batch in data:
                loss, metrics = self.train_step(batch)
                epoch_loss += loss

                if metrics is not None:
                    for metric_index, metric in enumerate(metrics):
                        metrics[metric_index] += metric

            epoch_loss.backward()
            self.optimizer.step()
            loss = epoch_loss.item() / len(data)
            training_history.append(loss)
            metric_history.append(metrics)
            pbar.set_description(f"Loss: {round(loss, 5)}")

        return training_history, metric_history

    def inference(self, data):
        losses = FeatureLoss("tracks").included_features.keys()
        active_loss = {loss: [] for loss in losses}

        for batch in tqdm(data, desc="Computing Feature Loss..."):
            m = self.probe.input_function(batch)
            prediction = self.probe.forward(m)

            for loss in losses:
                active_loss[loss].append(
                    FeatureLoss(loss, device=self.device)
                    .loss(prediction, batch)
                    .detach()
                    .cpu()
                )

        output = {
            loss_name: np.mean(np.array(loss_value)).astype(float)
            for loss_name, loss_value in active_loss.items()
        }
        return output
