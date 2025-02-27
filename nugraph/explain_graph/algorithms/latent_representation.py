from collections import defaultdict
import json
import pickle
from typing import Sequence
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA, IncrementalPCA
import h5py
from sklearn.metrics import silhouette_samples

from matplotlib import pyplot as plt
import nugraph.explain_graph.utils.visuals_common as plot_utils
import numpy as np
from tqdm import tqdm


class BatchedFit:
    def __init__(
        self,
        dataloader,
        transform,
        planes,
        embedding_function=None,
        max_features=30,
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

            batched_decomp[plane] = np.concatenate(batched_decomp[plane])

        return batched_decomp, self.pca


class LatentRepresentation:
    def __init__(
        self,
        embedding_function,
        data_loader,
        out_path,
        batched_fit: bool = True,
        name="plot",
        title="",
        true_labels=None,
        decomposition_algorithm="pca",
        planes: Sequence = ("u", "v", "y"),
        n_threshold: int = None,
    ) -> None:
        self.true_labels = true_labels
        self.embedding_function = embedding_function
        self.data_loader = data_loader
        self.planes = planes

        self.n_components = 30

        self.decomposition_algorithm = decomposition_algorithm
        self.batched_fit = batched_fit

        self.out_path = out_path
        self.plot_name = name
        self.title = title

        self.threshold = 0.02 if n_threshold is None else n_threshold
        self.random = np.random.default_rng()

        # Objects to save
        self.decomposition = {}
        self.fit = {}
        self.silhouette_summary = {}

    def subset(self, plane):
        if self.threshold < 1:
            sample_threshold = int(
                len(self.decomposition[self.planes[0]]) * self.threshold
            )
        else:
            sample_threshold = int(self.threshold)

        sample_index = self.random.integers(
            low=0, high=len(self.decomposition[plane]), size=sample_threshold
        )
        return self.decomposition[plane][sample_index], sample_index

    def decompose(self):
        try:
            decomp = {
                "isomap": Isomap(n_components=self.n_components, n_neighbors=5),
                "pca": PCA(n_components=self.n_components),
                "t_sne": TSNE(n_components=self.n_components),
            }[self.decomposition_algorithm]

        except KeyError:
            raise NotImplementedError(
                f"Algorithm {self.decomposition_algorithm} not included."
            )
        max_features = 30
        if self.batched_fit:
            fit = BatchedFit(
                self.data_loader,
                decomp,
                max_features=max_features,
                planes=self.planes,
                embedding_function=self.embedding_function,
            )
            fit.fit()
            decomposition, fits = fit.transform()

            self.decomposition = decomposition
            self.fit = fits

        else:
            self.non_batched_decomp(decomp)

        # Todo h5 obj
        with open(
            f"{self.out_path.strip('/')}/{self.plot_name}_decomposition.json", "w"
        ) as f:
            json.dump({}, f)

        return self.decomposition

    def non_batched_decomp(self, decomp):
        for plane in self.planes:
            data = []
            for batch in self.data_loader:
                embedding = self.embedding_function(batch)
                data.append(
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
            embedding = np.concatenate(data)
            fit = decomp.fit(embedding)
            self.fit[plane] = fit
            self.decomposition[plane] = decomp.transform(embedding).astype(np.float64)

    def visualize_label_silhouette(self):
        if self.decomposition is None:
            raise ValueError("No decomposition or embedding present")

        figure, subplots = plt.subplots(
            len(self.planes), 2, figsize=(10, 5 * len(self.planes))
        )  # cluster by plane
        plt.setp(subplots, xticks=[], yticks=[])
        index_map, handles = plot_utils.color_map()
        plt.figlegend(handles=handles)

        position = 10
        for subplot_index, plane in enumerate(self.planes):
            self.silhouette_summary[plane] = {}

            samples, sample_index = self.subset(plane)
            labels = np.array(
                [graph[plane]["y_semantic"].tolist() for graph in self.data_loader]
            ).ravel()[sample_index]

            plane_silhouette = silhouette_samples(
                self.decomposition[plane][sample_index], labels
            )

            subplots[subplot_index, 0].axvline(x=np.mean(plane_silhouette))
            subplots[subplot_index, 0].set_ylabel(plane)

            for label in set(labels):
                scores = plane_silhouette[labels == label]
                self.silhouette_summary[plane][f"{label}"] = float(np.mean(scores))
                scores.sort()
                y_upper = position + scores.shape[0]

                # Fill between for the scores
                subplots[subplot_index, 0].fill_betweenx(
                    np.arange(position, y_upper),
                    0,
                    scores,
                    alpha=0.7,
                    color=index_map[label][0],
                )

                # Move the index up
                position = y_upper + 10

                subplots[subplot_index, 1].scatter(
                    samples[:, 0][labels == label],
                    samples[:, 1][labels == label],
                    color=index_map[label][0],
                    marker=index_map[label][1],
                )

        if self.title != "":
            self.title += ":"

        figure.suptitle(f"{self.title}")
        plt.savefig(f"{self.out_path}/{self.plot_name}.png")
        plt.close()

    def save_results(self):
        # write the fit, embedded fit, and silhouette results
        with h5py.File(
            f"{self.out_path.rstrip('/')}/{self.plot_name}_decomp.h5", "w"
        ) as f:
            f.create_group("decomposition")
            for key, array in self.decomposition.items():
                f["decomposition"].create_dataset(key, data=array, dtype=np.float64)

        with open(f"{self.out_path.rstrip('/')}/{self.plot_name}_fit.pkl", "wb") as f:
            pickle.dump([i for i in self.fit.values()], f)

        with open(
            f"{self.out_path.rstrip('/')}/{self.plot_name}_summary.json", "w"
        ) as f:
            json.dump(self.silhouette_summary, f)
