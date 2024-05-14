from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

from sklearn.metrics import silhouette_samples

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class LatentRepresentation:
    def __init__(
        self,
        weights,
        out_path,
        name="plot",
        title="",
        n_dim=2,
        n_clusters=5,
        true_labels=None,
        decomposition_algorithm="isomap",
        clustering_algorithm="kmeans",
    ) -> None:
        self.weights = weights
        self.planes = weights.keys()
        self.true_labels = true_labels

        self.n_components = n_dim
        self.n_clusters = n_clusters

        self.decomposition_algorithm = decomposition_algorithm
        self.clustering_algorithm = clustering_algorithm

        self.decomposition = None
        self.clustering_labels = None
        self.clustering_score = None

        self.out_path = out_path
        self.plot_name = name
        self.title = title

    def decompose(self):
        try:
            decomp = {
                "isomap": Isomap(
                    n_components=self.n_components, n_neighbors=self.n_components
                ).fit,
                "pca": PCA(n_components=self.n_components).fit,
                "t_sne": TSNE(n_components=self.n_components).fit,
            }[self.decomposition_algorithm]
        except KeyError:
            raise NotImplementedError(
                f"Algorithm {self.decomposition_algorithm} not included."
            )

        all_weights = np.concatenate([self.weights[p] for p in self.planes])
        decomposition_engine = decomp(all_weights)
        self.decomposition = {
            plane: decomposition_engine.transform(self.weights[plane])
            for plane in self.planes
        }

    def cluster(self):
        try:
            clustering_algo = {
                "kmeans": KMeans(n_clusters=self.n_clusters).fit_transform,
                "dbscan": DBSCAN().fit_predict,
            }[self.clustering_algorithm]
        except KeyError:
            raise NotImplementedError("Clustering algorithm not implemented")

        if self.decomposition is None:
            self.decompose()

        self.clustering_labels = {
            plane: np.argmax(clustering_algo(self.decomposition[plane]), axis=-1)
            for plane in self.planes
        }
        self.clustering_score = self.score()

    def score(self):
        return {
            plane: silhouette_samples(
                self.decomposition[plane], self.clustering_labels[plane]
            )
            for plane in self.planes
        }

    def visualize(self):
        if self.n_components != 2:
            raise ValueError(
                "Cannot make visualization for more dimensions other than 2."
            )

        if self.decomposition is None:
            self.decompose()

        if self.clustering_labels is None:
            self.cluster()

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
                color_map = {
                    index: color for index, color in zip(labels_classes, colors)
                }
                label_indices = [-1] + [i for i in range(len(labels_classes))]
                index_map = {
                    index: color for index, color in zip(label_indices, colors)
                }
                handles = [
                    mpatches.Patch(color=color, label=label)
                    for label, color in color_map.items()
                ]

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

        figure.suptitle(
            f"{self.title} Clustering using {self.clustering_algorithm} and {self.decomposition_algorithm} Decomposition"
        )
        if n_columns == 3:
            subplots[-1, 2].set_xlabel("True Labels")
        plt.savefig(f"{self.out_path}/{self.plot_name}.png")
        plt.close()

    def save(self):
        ""

    def __call__(self):
        self.visualize()
        self.save()
