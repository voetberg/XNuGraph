from collections import defaultdict
import json
from typing import Sequence
from sklearn.manifold import Isomap, TSNE

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans

from sklearn.metrics import silhouette_samples

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm


class BatchedFit:
    def __init__(
        self,
        dataloader,
        transform,
        planes,
        embedding_function=None,
        max_features=20,
        batch_size=200,
    ) -> None:
        self.batch_size = batch_size
        self.planes = planes
        self.pca = {}
        self.max_features = max_features
        self.batch_size = batch_size
        self.loader = dataloader
        self.transform_function = transform
        self.trained_transfrom = {}
        self.embedding_function = embedding_function

    def fit(self):
        # Following the recommendation of
        # the original author and refusing it down via PCA first
        # PCA Can handle batching

        for plane in self.planes:
            pca = IncrementalPCA(
                n_components=self.max_features, batch_size=self.batch_size
            )
            for batch_index, batch in enumerate(
                tqdm(self.loader, desc=f"Training PCA for Plane {plane}...")
            ):
                if batch_index > 0.25 * len(self.loader):
                    continue
                # get the embedding of the batch
                embedding = self.embedding_function(batch)[plane].detach()

                if embedding.shape[0] != 0:
                    ravelled = embedding.reshape(
                        (embedding.shape[0], embedding.shape[1] * embedding.shape[2])
                    ).cpu()
                    pca = pca.partial_fit(ravelled)

            self.pca[plane] = pca

    def transform(self):
        batched_decomp = defaultdict(list)
        batched_embedding = defaultdict(list)

        for plane in self.planes:
            for batch_index, batch in enumerate(
                tqdm(self.loader, desc=f"Transforming PCA for Plane {plane}...")
            ):
                if batch_index > 0.25 * len(self.loader):
                    continue

                embedding = self.embedding_function(batch)
                ravelled = (
                    embedding[plane]
                    .reshape(
                        (
                            embedding[plane].shape[0],
                            embedding[plane].shape[1] * embedding[plane].shape[2],
                        )
                    )
                    .detach()
                    .cpu()
                )
                decomp = self.pca[plane].transform(ravelled)
                batched_decomp[plane].append(decomp)
                batched_embedding[plane].append(ravelled)

            batched_decomp[plane] = np.concatenate(batched_decomp[plane])
            batched_embedding[plane] = np.concatenate(batched_embedding[plane])

        return batched_decomp, batched_embedding


class BatchedClustering:
    def __init__(self, n_clusters, batch_size=200) -> None:
        self.n_clusters = n_clusters
        self.batch_size = batch_size

    def fit(self, decomposition):
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=0,
            batch_size=self.batch_size,
            n_init="auto",
        )
        n_batches = int(len(decomposition) / self.batch_size)
        decomp_batches = np.array_split(decomposition, n_batches, axis=0)
        for batch in tqdm(decomp_batches, desc="Training KMeans..."):
            kmeans = kmeans.partial_fit(batch)
        self.kmeans = kmeans

    def transform(self, decomposition):
        labels = []

        n_batches = int(len(decomposition) / self.batch_size)
        decomp_batches = np.array_split(decomposition, n_batches, axis=0)
        for batch in decomp_batches:
            batch_labels = self.kmeans.predict(batch)
            labels.append(batch_labels)

        return np.concatenate(labels)

    def fit_transform(self, decomposition):
        self.fit(decomposition)
        return self.transform(decomposition)


class LatentRepresentation:
    def __init__(
        self,
        embedding_function,
        data_loader,
        out_path,
        batched_fit: bool = True,
        batched_cluster: bool = True,
        name="plot",
        title="",
        n_visual_dim=2,
        n_clusters=5,
        clustering_components: int = 2,
        true_labels=None,
        decomposition_algorithm="kmeans",
        clustering_algorithm="batched_kmeans",
        planes: Sequence = ("u", "v", "y"),
        n_threshold: int = None,
    ) -> None:
        self.true_labels = true_labels
        self.embedding_function = embedding_function
        self.data_loader = data_loader
        self.planes = planes

        self.n_clustering_components = clustering_components
        self.n_components = n_visual_dim
        self.n_clusters = n_clusters

        self.decomposition_algorithm = decomposition_algorithm
        self.clustering_algorithm = clustering_algorithm
        self.batched_fit = batched_fit
        self.batched_cluster = batched_cluster

        self.decomposition = None
        self.clustering_labels = None
        self.clustering_score = None

        self.out_path = out_path
        self.plot_name = name
        self.title = title

        self.threshold = 0.02 if n_threshold is None else n_threshold
        self.random = np.random.default_rng()

    def subset(self, plane):
        if not isinstance(self.threshold, int):
            sample_threshold = int(
                len(self.decomposition[self.planes[0]]) * self.threshold
            )
        else:
            sample_threshold = self.threshold

        sample_index = self.random.integers(
            low=0, high=len(self.decomposition[plane]), size=sample_threshold
        )

        return self.decomposition[plane][sample_index], sample_index

    def decompose(self, clustering_decomp: bool = True):
        if clustering_decomp:
            n_components = self.n_clustering_components
        else:
            n_components = self.n_components

        if self.batched_cluster:
            decomp = None
            max_features = n_components

        else:
            try:
                decomp = {
                    "isomap": Isomap(
                        n_components=n_components, n_neighbors=self.n_clusters
                    ),
                    "pca": PCA(n_components=n_components),
                    "t_sne": TSNE(n_components=n_components),
                }[self.decomposition_algorithm]

            except KeyError:
                raise NotImplementedError(
                    f"Algorithm {self.decomposition_algorithm} not included."
                )
            max_features = 30

        fit = BatchedFit(
            self.data_loader,
            decomp,
            max_features=max_features,
            planes=self.planes,
            embedding_function=self.embedding_function,
        )

        fit.fit()
        decomposition, embedding = fit.transform()

        results = {
            "decomposition": {
                key: decomp.tolist() for key, decomp in decomposition.items()
            },
            "embedding": {
                key: embedding.tolist() for key, embedding in embedding.items()
            },
        }

        with open(
            f"{self.out_path.strip('/')}/{self.plot_name}_decomposition.json", "w"
        ) as f:
            json.dump(results, f)

        self.decomposition = decomposition
        self.embedding = embedding

        return decomposition

    def color_map(self):
        colors = [
            "grey",
            "darkorange",
            "dodgerblue",
            "limegreen",
            "palevioletred",
            "indigo",
        ]
        labels_classes = [
            "Background",
            "MIP",
            "HIP",
            "shower",
            "michel",
            "diffuse",
        ]
        color_map = {index: color for index, color in zip(labels_classes, colors)}
        label_indices = [-1] + [i for i in range(len(labels_classes))]
        index_map = {index: color for index, color in zip(label_indices, colors)}
        handles = [
            mpatches.Patch(color=color, label=label)
            for label, color in color_map.items()
        ]
        return index_map, handles

    def cluster(self):
        try:
            clustering_algo = {
                "kmeans": KMeans(n_clusters=self.n_clusters),
                "batched_kmeans": BatchedClustering(n_clusters=self.n_clusters),
                "dbscan": DBSCAN(),
            }[self.clustering_algorithm]
        except KeyError:
            raise NotImplementedError("Clustering algorithm not implemented")

        if self.decomposition is None:
            self.decomposition = self.decompose(clustering_decomp=True)

        self.clustering_labels = {}
        self.clustering_score = {}
        results = {}
        for plane in self.planes:
            clustering_algo.fit(self.decomposition[plane])

            samples, sample_index = self.subset(plane)
            labels = clustering_algo.transform(samples)

            print(f"Computing Silhouette for {len(labels)} samples.....")
            self.clustering_labels[plane] = labels
            self.clustering_score[plane] = silhouette_samples(samples, labels)
            self.decomposition[plane] = samples
            self.true_labels[plane] = self.true_labels[plane][sample_index]

        results = {
            "cluster_labels": self.clustering_labels,
            "cluster_score": self.clustering_score,
        }
        with open(
            f"{self.out_path.strip('/')}/{self.plot_name}_clusters.json", "w"
        ) as f:
            json.dump(results, f)

    def visualize_clusters(self):
        if self.decomposition is None:
            raise ValueError("No decomposition present, run self.decompose")

        if self.clustering_labels is None:
            raise ValueError("No labels present, run self.cluster")
        if self.n_components != 2:
            raise ValueError(
                "Cannot make visualization for more dimensions other than 2."
            )

        n_columns = 2 if self.true_labels is None else 3
        figure, subplots = plt.subplots(
            len(self.planes), n_columns, figsize=(4 * n_columns, 5 * len(self.planes))
        )  # cluster by plane
        plt.setp(subplots, xticks=[], yticks=[])

        position = 10
        for subplot_index, plane in enumerate(self.planes):
            position = 10
            subplots[subplot_index, 0].axvline(x=np.mean(self.clustering_score[plane]))
            for label in set(self.clustering_labels[plane]):
                subplots[subplot_index, 0].set_ylabel(plane)
                cluster_scores = self.clustering_score[plane][
                    self.clustering_labels[plane] == label
                ]
                cluster_scores.sort()
                y_upper = position + cluster_scores.shape[0]

                # Fill between for the scores
                subplots[subplot_index, 0].fill_betweenx(
                    np.arange(position, y_upper),
                    0,
                    cluster_scores,
                    alpha=0.7,
                )

                # Move the index up
                position = y_upper + 10

                # get the position of the cluster
                decomp_position = self.decomposition[plane][
                    self.clustering_labels[plane] == label
                ]
                subplots[subplot_index, 1].scatter(
                    decomp_position[:, 0], decomp_position[:, 1]
                )

            if n_columns == 3:
                index_map, handles = self.color_map()

                subplots[subplot_index, 2].scatter(
                    self.decomposition[plane][:, 0],
                    self.decomposition[plane][:, 1],
                    c=[index_map[label.item()] for label in self.true_labels[plane]],
                )
                subplots[0, 2].legend(handles=handles)

        subplots[-1, 0].set_xlabel("Silhouette scores of clusters")
        subplots[-1, 1].set_xlabel("Clustering in decomposition space")

        if self.title != "":
            self.title += ":"

        figure.suptitle(f"{self.title}")
        if n_columns == 3:
            subplots[-1, 2].set_xlabel("True Labels")
        plt.savefig(f"{self.out_path}/{self.plot_name}.png")
        plt.close()

    def visualize_label_silhouette(self):
        if (self.decomposition is None) or (self.embedding is None):
            raise ValueError(
                "No decomposition or embedding present, run self.decomposition"
            )

        figure, subplots = plt.subplots(
            len(self.planes), 2, figsize=(8, 5 * len(self.planes))
        )  # cluster by plane
        plt.setp(subplots, xticks=[], yticks=[])

        index_map, handles = self.color_map()
        subplots[0, 1].legend(handles=handles)

        position = 10
        for subplot_index, plane in enumerate(self.planes):
            samples, sample_index = self.subset(plane)
            labels = np.array(
                [graph[plane]["y_semantic"].tolist() for graph in self.data_loader]
            ).ravel()[sample_index]
            plane_silhouette = silhouette_samples(
                self.embedding[plane][sample_index], labels
            )

            subplots[subplot_index, 0].axvline(x=np.mean(plane_silhouette))
            subplots[subplot_index, 0].set_ylabel(plane)

            for label in set(self.true_labels[plane]):
                scores = plane_silhouette[labels == label]
                scores.sort()
                y_upper = position + scores.shape[0]

                # Fill between for the scores
                subplots[subplot_index, 0].fill_betweenx(
                    np.arange(position, y_upper),
                    0,
                    scores,
                    alpha=0.7,
                    color=index_map[label],
                )

                # Move the index up
                position = y_upper + 10

            subplots[subplot_index, 1].scatter(
                samples[:, 0],
                samples[:, 1],
                c=[index_map[label.item()] for label in labels],
            )

        if self.title != "":
            self.title += ":"

        figure.suptitle(f"{self.title}")
        plt.savefig(f"{self.out_path}/{self.plot_name}.png")
        plt.close()

    def __call__(self):
        self.n_components = 2
        self.decomposition = self.decompose(clustering_decomp=True)

        if self.clustering_labels is None:
            self.cluster()

        self.visualize_clusters()
