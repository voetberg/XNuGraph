from collections import defaultdict
import imageio
import os
from typing import Literal, Sequence, Optional
from scipy.spatial import Delaunay
from torch_geometric.data import HeteroData
import numpy as np
import torch
from tqdm import trange
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals
from nugraph.explain_graph.algorithms.linear_probes.graph_priors import (
    RandomPrior,
    NormalPrior,
    MeanDifferencePrior,
    MeanPrior,
)
import matplotlib.pyplot as plt


class ActivatedVector:
    def __init__(
        self,
        trained_probe,
        initialization_scheme: Literal[
            "random", "normal", "mean", "mean_difference", "sample"
        ] = "random",
        outdir="./",
        data_path=None,
        planes: Sequence[str] = ("u", "v", "y"),
        range_planar_nodes: tuple[float] = (75, 300),
        range_nexus_nodes: tuple[float] = (20, 75),
        position_range: dict[str, tuple[tuple]] = {
            "u": ((545, 555), (200, 300)),
            "v": ((440, 475), (200, 300)),
            "y": ((715, 860), (200, 300)),
        },
        l2: bool = False,
        norm_clip: bool = False,
        l2_decay: float = 0.01,
        norm_threshold: float = 0.1,
        feature_range: tuple = (-2, 2),
        n_features: int = 6,
        device: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.outdir = outdir
        if not os.path.exists(os.path.dirname(outdir)):
            os.makedirs(os.path.dirname(outdir))

        self.rng = np.random.default_rng(random_seed)
        self.probe = trained_probe
        self.data_path = data_path

        self.planes = planes
        self.position_range = position_range
        self.feature_range = feature_range
        self.n_features = n_features
        assert set(planes) == set(
            position_range.keys()
        ), "Position range must have planes as keys"

        self.device = torch.device("cpu") if device is None else device
        self.graph = self.init_graph(
            range_planar_nodes, range_nexus_nodes, initialization_scheme
        )
        self.mean_gradient_difference = defaultdict(lambda: [0])

        self.l2 = l2
        self.l2_decay = l2_decay
        self.norm_clip = norm_clip
        self.norm_threshold = norm_threshold

    def normalize_position(self, planar_positions, plane):
        return torch.stack(
            (
                (self.position_range[plane][0][1] - self.position_range[plane][0][0])
                * planar_positions[:, 0]
                + self.position_range[plane][0][1],
                (self.position_range[plane][1][1] - self.position_range[plane][1][0])
                * planar_positions[:, 1]
                + self.position_range[plane][1][1],
            )
        ).T

    def make_edges(self, graph, nexus_positions):
        edges = {}
        for plane in self.planes:
            positions = self.normalize_position(
                planar_positions=graph[plane]["x"][:, 2:], plane=plane
            )
            calced_edges = Delaunay(positions.detach()).simplices[:, :-1]

            abs_nexus_position = torch.sum(nexus_positions, -1) / 2
            abs_position = torch.sum(positions.detach(), -1) / 2

            if len(abs_position) > len(abs_nexus_position):
                abs_position = abs_position[
                    torch.randperm(len(abs_position))[: len(abs_nexus_position)]
                ]
            else:
                abs_nexus_position = abs_nexus_position[
                    torch.randperm(len(abs_nexus_position))[: len(abs_position)]
                ]

            nexus_edges = Delaunay(
                np.stack((abs_position, abs_nexus_position)).T
            ).simplices[:, :-1]

            edges[(plane, "plane", plane)] = {
                "edge_index": torch.tensor(
                    calced_edges.T, device=self.device, dtype=torch.int64
                )
            }
            edges[(plane, "nexus", "sp")] = {
                "edge_index": torch.tensor(
                    nexus_edges.T, device=self.device, dtype=torch.int64
                )
            }

        return edges

    def _pick_init(self, init, n_planar_range, n_nexus_range):
        if init == "sample":
            raise NotImplementedError
        elif init == "normal":
            scheme = NormalPrior(
                self.planes, None, None, self.rng, self.device, data_path=self.data_path
            )
        elif init == "mean":
            scheme = MeanPrior(
                self.planes,
                None,
                None,
                rng=self.rng,
                device=self.device,
                data_path=self.data_path,
            )
        elif init == "mean_difference":
            scheme = MeanDifferencePrior(
                self.planes,
                n_planar_range,
                n_nexus_range,
                self.rng,
                device=self.device,
                data_path=self.data_path,
            )
        elif init == "random":
            scheme = RandomPrior(
                self.planes,
                n_planar_range,
                n_nexus_range,
                self.rng,
                device=self.device,
                feature_range=self.feature_range,
                n_features=4,
            )
        else:
            raise NotImplementedError(
                f"Initialization scheme {scheme} not included. Choose from (sample, normal, mean, mean_difference, random)"
            )
        return scheme

    def init_graph(self, n_planar_range, n_nexus_range, initialization_scheme):
        scheme = self._pick_init(initialization_scheme, n_planar_range, n_nexus_range)
        graph, nexus_positions = scheme.prior()
        self.nexus_positions = nexus_positions
        edges = self.make_edges(graph, nexus_positions=self.nexus_positions)

        graph.update(edges)
        graph = HeteroData(graph)
        return graph

    def forward_probe(self):
        m = self.probe.input_function(self.graph)
        prediction = self.probe.forward(m)
        return prediction

    def get_activation(self):
        gradient = {}
        prediction = self.forward_probe()
        for plane in self.planes:
            # compute gradients w.r.t. target unit,
            # then access the gradient of input (image) w.r.t. target unit (neuron)
            prediction[plane].retain_grad()
            self.graph.collect("x")[plane].retain_grad()
            prediction[plane].backward(
                gradient=torch.ones_like(prediction[plane]), retain_graph=True
            )
            gradient[plane] = self.graph.collect("x")[plane].grad[:, :4]
        return gradient

    def calculate_activation_step(self, alpha):
        # Propagate image through network,
        # then access activation of target layer
        gradient = self.get_activation()
        for plane in self.planes:
            self.mean_gradient_difference[plane].append(
                abs(
                    self.mean_gradient_difference[plane][-1]
                    - torch.mean(gradient[plane])
                )
            )
        self.mutate_graph(gradient, alpha)

    def clip(self, array, threshold):
        norm = torch.norm(array, dim=0)
        norm = norm.numpy()

        # Create a binary matrix, with 1's wherever the pixel falls below threshold
        smalls = norm < np.percentile(norm, threshold)

        # Crop pixels from image
        crop = array - array * smalls
        return crop.unsqueeze(0)

    def mutate_graph(self, gradient, alpha):
        # Gradient Step
        # input = input + alpha * d_graph/d_neuron
        node_features = self.graph.collect("x")
        for plane in self.planes:
            # Move the features
            features = torch.add(
                node_features[plane][:, 0:4], torch.mul(gradient[plane], alpha)
            )

            with torch.no_grad():
                # Regularization: L2
                if self.l2:
                    features = torch.mul(features, (1.0 - self.l2_decay))

                # Regularization: Clip Norm
                if self.norm_clip:
                    features = self.calculate_activation_step(
                        features.detach().squeeze(0), threshold=self.norm_threshold
                    )

            features.requires_grad_(True)
            self.graph[plane]["x"] = features

        # Recalculate edges
        edges = self.make_edges(self.graph, self.nexus_positions)
        for plane in self.planes:
            self.graph[(plane, "plane", plane)]["edge_index"] = edges[
                (plane, "plane", plane)
            ]["edge_index"]
            self.graph[(plane, "nexus", "sp")]["edge_index"] = edges[
                (plane, "nexus", "sp")
            ]["edge_index"]

    def visualize_graph(self, plot_index=None):
        name = "graph_maximize.png"
        if plot_index is not None:
            name = f"_{name.rstrip('.png')}_step_{plot_index}.png"

        for plane in self.planes:
            self.graph[plane]["pos"] = self.normalize_position(
                self.graph[plane]["x"][:, 2:], plane
            )
        EdgeVisuals(planes=self.planes).plot(
            self.graph,
            outdir=self.outdir,
            file_name=name,
        )
        plt.close("all")

    def visualize_graph_labels(self, model, graph=None):
        graph = graph if graph is not None else self.graph
        name = "graph_labeled.png"
        prediction = model.forward(*model.unpack_batch(graph))
        for plane in self.planes:
            background = prediction["x_filter"][plane] < 0.5
            filtered = torch.where(
                background,
                -1,
                torch.argmax(prediction["x_semantic"][plane].detach(), axis=1),
            )
            graph[plane]["y_semantic"] = filtered
            graph[plane]["x_semantic"] = prediction["x_semantic"][plane].detach()
            graph[plane]["pos"] = self.normalize_position(
                graph[plane]["x"][:, 2:], plane
            ).detach()

        EdgeVisuals(planes=self.planes).event_plot(
            graph,
            outdir=self.outdir,
            file_name=name,
        )

    def _history_gif(self):
        out_file = f"{self.outdir.rstrip('/')}/graph_maximize_history.gif"
        filenames = [
            f"{self.outdir.rstrip('/')}/{f}"
            for f in os.listdir(self.outdir)
            if "_graph_maximize_step_" in f
        ]
        with imageio.get_writer(out_file, mode="I") as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                for _ in range(20):
                    writer.append_data(image)

        for f in filenames:
            os.remove(f)

    def visualize_history(self):
        figure, subplot = plt.subplots()
        for plane in self.planes:
            grad_history = self.mean_gradient_difference[plane]
            steps = range(len(grad_history))
            subplot.plot(steps, grad_history, label=plane)

        figure.legend()
        figure.savefig(f"{self.outdir.rstrip('/')}/gradient_history.png")

    def __call__(
        self,
        steps: int = 5,
        gradient_alpha: float = 0.3,
        visualize_each_step: bool = False,
    ):
        # Based on this code: https://github.com/Nguyen-Hoa/Activation-Maximization/tree/master
        for idx in trange(steps):
            self.calculate_activation_step(gradient_alpha)
            if visualize_each_step:
                self.visualize_graph(plot_index=idx + 1)

        if visualize_each_step:
            self._history_gif()

        self.visualize_history()
        return self.graph
