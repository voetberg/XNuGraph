from collections import defaultdict
import numpy as np
import torch

from nugraph.explain_graph.utils import Load


class GraphPrior:
    def __init__(self, planes, range_planar_nodes, range_nexus_nodes) -> None:
        self.planes = planes
        self.range_planar_nodes = range_planar_nodes
        self.range_nexus_nodes = range_nexus_nodes

    def make_planar_hits(self, n_hits, plane):
        raise NotImplementedError

    def make_nexus_hits(self, n_hits):
        raise NotImplementedError

    def n_hits(self):
        raise NotImplementedError

    def prior(self):
        graph = {}
        # Random points for each plane
        n_planar, n_nexus = self.n_hits()

        for plane in self.planes:
            graph[plane] = self.make_planar_hits(n_planar[plane], plane)

        graph["sp"] = {"num_nodes": n_nexus}
        nexus_positions = self.make_nexus_hits(n_nexus)
        return graph, nexus_positions


class RandomPrior(GraphPrior):
    def __init__(
        self,
        planes,
        range_planar_nodes,
        range_nexus_nodes,
        rng,
        device,
        feature_range,
        n_features,
    ) -> None:
        super().__init__(planes, range_planar_nodes, range_nexus_nodes)
        self.feature_range = feature_range
        self.n_features = n_features
        self.rng = rng
        self.device = device

    def make_planar_hits(self, n_hits, plane):
        x_features = torch.tensor(
            self.rng.uniform(*self.feature_range, size=(n_hits, self.n_features)),
            device=self.device,
            dtype=torch.float,
            requires_grad=True,
        )

        planar = {
            "x": x_features,
            "id": torch.tensor(
                np.linspace(0, n_hits).astype(int),
                device=self.device,
                dtype=torch.double,
            ),
            "batch": torch.tensor(
                np.zeros(shape=(n_hits)), device=self.device, dtype=torch.float
            ),
        }
        return planar

    def make_nexus_hits(self, n_hits):
        return torch.tensor(self.rng.uniform(size=(n_hits, 2)), device=self.device)

    def n_hits(self):
        n_planar_hits = {
            plane: self.rng.integers(*self.range_planar_nodes) for plane in self.planes
        }
        n_nexus_hits = self.rng.integers(*self.range_nexus_nodes)
        return n_planar_hits, n_nexus_hits


class NormalPrior(GraphPrior):
    # Draw from a normal distribution matching the data
    def __init__(
        self, planes, range_planar_nodes, range_nexus_nodes, rng, device, data_path
    ) -> None:
        super().__init__(planes, range_planar_nodes, range_nexus_nodes)
        self.rng = rng
        self.device = device

        self._load_dataset(data_path)
        self._calculate_mean()
        self._calculate_variance()

    def _load_dataset(self, path):
        data = Load(data_path=path, planes=self.planes).data

        self.features = defaultdict(lambda: defaultdict(list))
        for plane in self.planes:
            for graph in data:
                self.features[plane]["x"] += graph[plane]["x"].tolist()
                self.n_features = graph[plane]["x"].shape[1]
                count = graph[plane]["x"].shape[0]
                if count != 0:
                    self.features[plane]["count"].append(count)
                nexus_edges = graph[(plane, "nexus", "sp")]["edge_index"]
                if nexus_edges.shape[-1] != 0:
                    self.features["sp"]["count"].append(
                        torch.unique(nexus_edges[1]).size()[0]
                    )

    def _calculate_mean(self):
        self.mean_features = defaultdict()
        self.mean_hits = defaultdict()

        for plane in self.planes:
            self.mean_hits[plane] = int(np.mean(self.features[plane]["count"]))
            n_drop = len(self.features[plane]["x"]) % self.n_features
            reshaped = np.array(self.features[plane]["x"][: -1 * n_drop]).reshape(
                (-1, self.n_features)
            )
            self.mean_features[plane] = np.mean(reshaped, axis=0)
        self.mean_nexus_hits = np.mean(self.features["sp"]["count"])

    def _calculate_variance(self):
        self.variance_features = defaultdict()
        self.variance_hits = defaultdict()

        for plane in self.planes:
            self.variance_hits[plane] = int(np.std(self.features[plane]["count"]))
            n_drop = len(self.features[plane]["x"]) % self.n_features
            reshaped = np.array(self.features[plane]["x"][: -1 * n_drop]).reshape(
                (-1, self.n_features)
            )
            deviation = np.std(reshaped, axis=0)
            for index, val in enumerate(deviation):
                if val < 10 ** (-3):
                    deviation[index] = 0.1  # Terrible fix. Don't do this.
            self.variance_features[plane] = deviation
        self.variance_nexus_hits = np.std(self.features["sp"]["count"])

    def n_hits(self):
        plane_hits = self.mean_hits.copy()
        for plane in self.planes:
            hit = int(
                self.rng.normal(
                    loc=self.mean_hits[plane], scale=self.variance_hits[plane]
                )
            )
            if hit > 0:
                plane_hits[plane] = hit

        nexus_hits = abs(
            int(
                self.rng.normal(
                    loc=self.mean_nexus_hits, scale=self.variance_nexus_hits
                )
            )
        )
        return plane_hits, nexus_hits

    def make_planar_hits(self, n_hits, plane):
        x_features = torch.tensor(
            self.rng.normal(
                loc=self.mean_features[plane],
                scale=self.variance_features[plane],
                size=(n_hits, self.n_features),
            ),
            device=self.device,
            dtype=torch.float,
            requires_grad=True,
        )

        planar = {
            "x": x_features,
            "id": torch.tensor(
                np.linspace(0, n_hits).astype(int),
                device=self.device,
                dtype=torch.double,
            ),
            "batch": torch.tensor(
                np.zeros(shape=(n_hits)), device=self.device, dtype=torch.float
            ),
        }
        return planar

    def make_nexus_hits(self, n_hits):
        return torch.tensor(
            self.rng.normal(
                loc=self.mean_nexus_hits,
                scale=self.variance_nexus_hits,
                size=(n_hits, 2),
            ),
            device=self.device,
        )


class MeanPrior(GraphPrior):
    def __init__(
        self, planes, range_planar_nodes, range_nexus_nodes, rng, device, data_path
    ) -> None:
        super().__init__(planes, range_planar_nodes, range_nexus_nodes)
        self.rng = rng
        self.device = device

        self._load_dataset(data_path)
        self._calculate_mean()

    def _load_dataset(self, path):
        data = Load(planes=self.planes, auto_load=False).load_data(path)

        self.features = defaultdict(lambda: defaultdict(list))
        for plane in self.planes:
            for graph in data:
                self.features[plane]["x"] += graph[plane]["x"].tolist()
                self.n_features = graph[plane]["x"].shape[1]
                count = graph[plane]["x"].shape[0]
                if count != 0:
                    self.features[plane]["count"].append(count)
                nexus_edges = graph[(plane, "nexus", "sp")]["edge_index"]
                if nexus_edges.shape[-1] != 0:
                    self.features["sp"]["count"].append(
                        torch.unique(nexus_edges[1]).size()[0]
                    )

    def _calculate_mean(self):
        self.mean_features = defaultdict()
        self.mean_hits = defaultdict()

        for plane in self.planes:
            self.mean_hits[plane] = int(np.mean(self.features[plane]["count"]))
            n_drop = len(self.features[plane]["x"]) % self.n_features
            reshaped = np.array(self.features[plane]["x"][: -1 * n_drop]).reshape(
                (-1, self.n_features)
            )
            self.mean_features[plane] = np.mean(reshaped, axis=0)
        self.mean_nexus_hits = np.mean(self.features["sp"]["count"])

    def n_hits(self):
        return self.mean_hits, int(self.mean_nexus_hits)

    def make_planar_hits(self, n_hits, plane):
        x = self.rng.choice(
            self.mean_features[plane], replace=True, size=(n_hits, self.n_features)
        )
        x = torch.tensor(x, device=self.device, dtype=torch.float, requires_grad=True)

        planar = {
            "x": x,
            "id": torch.tensor(
                np.linspace(0, n_hits).astype(int),
                device=self.device,
                dtype=torch.double,
            ),
            "batch": torch.tensor(
                np.zeros(shape=(n_hits)), device=self.device, dtype=torch.float
            ),
        }
        return planar

    def make_nexus_hits(self, n_hits):
        return torch.tensor(self.rng.uniform(size=(n_hits, 2)), device=self.device)


class MeanDifferencePrior(GraphPrior):
    def __init__(
        self, planes, range_planar_nodes, range_nexus_nodes, rng, device, data_path
    ) -> None:
        super().__init__(planes, range_planar_nodes, range_nexus_nodes)
        self.mean_prior = MeanPrior(
            planes, range_planar_nodes, range_nexus_nodes, rng, device, data_path
        )
        self.random_prior = RandomPrior(
            planes,
            range_planar_nodes,
            range_nexus_nodes,
            rng,
            device,
            (-1, 1),
            n_features=self.mean_prior.n_features,
        )
        self.device = device
        self.rng = rng

    def n_hits(self):
        mean_hits = self.mean_prior.mean_hits
        mean_nexus_hits = self.mean_prior.mean_nexus_hits

        random_hits, random_nexus_hits = self.random_prior.n_hits()
        hits = {
            plane: abs(mean_hits[plane] - random_hits[plane]) for plane in self.planes
        }
        return hits, abs(int(mean_nexus_hits - random_nexus_hits))

    def make_planar_hits(self, n_hits, plane):
        x_mean = self.mean_prior.make_planar_hits(n_hits, plane)["x"]
        x_random = self.random_prior.make_planar_hits(n_hits, plane)["x"]
        planar = {
            "x": x_random - x_mean,
            "id": torch.tensor(
                np.linspace(0, n_hits).astype(int),
                device=self.device,
                dtype=torch.double,
            ),
            "batch": torch.tensor(
                np.zeros(shape=(n_hits)), device=self.device, dtype=torch.float
            ),
        }
        return planar

    def make_nexus_hits(self, n_hits):
        random = self.random_prior.make_nexus_hits(n_hits)
        mean = self.mean_prior.make_nexus_hits(n_hits)
        return random - mean


class SamplePrior(GraphPrior):
    # Use a random sample as the prior
    pass


class RandomSamplePrior(GraphPrior):
    # Combine different samples into a franken-prior
    pass
