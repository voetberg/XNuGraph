from typing import Sequence
from sklearn.manifold import Isomap, TSNE

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import DBSCAN, KMeans

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
        embedding_function,
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
            for batch_index, batch in enumerate(tqdm(self.loader, desc=f'Training PCA for Plane {plane}...')):

                # get the embedding of the batch
                embedding = self.embedding_function(batch)[plane].detach()

                if embedding.shape[0] != 0:
                    ravelled = embedding.reshape(
                        (embedding.shape[0], embedding.shape[1] * embedding.shape[2])
                    ).cpu()
                    pca = pca.partial_fit(ravelled)

            self.pca[plane] = pca

            # Then fit the actual transform 
            decomp = []
            for batch_index, batch in enumerate(tqdm(self.loader, desc=f'Training Decomposition for Plane {plane}...')): 
                if batch_index<= 0.25*len(self.loader): # Only uses a subset here 

                    embedding = self.embedding_function(batch)

                    ravelled = embedding[plane].reshape(
                                (embedding[plane].shape[0], embedding[plane].shape[1] * embedding[plane].shape[2])
                            ).detach().cpu()
                    
                    decomp.append(self.pca[plane].transform(ravelled))
            
            self.trained_transfrom[plane] = self.transform_function.fit(np.concatenate(decomp))

    def transform(self):
        batched_decomp = {plane:[] for plane in self.planes}

        for batch in tqdm(self.loader, desc="Transforming..."): 

            embedding = self.embedding_function(batch)

            for plane in self.planes: 
                ravelled = embedding.reshape(
                        (embedding[plane].shape[0], embedding[plane].shape[1] * embedding[plane].shape[2])
                    ).detach().cpu()
                decomp = self.pca[plane].transform(ravelled)
                batched_decomp[plane].append(self.trained_transfrom[plane].tranform(decomp))

            batched_decomp[plane] = np.concatenate(batched_decomp[plane])

        return batched_decomp


class LatentRepresentation:
    def __init__(
        self,
        embedding_function,
        data_loader,
        out_path,
        batched_fit: bool = True,
        name="plot",
        title="",
        n_visual_dim=2,
        n_clusters=5,
        clustering_components: int = 2,
        true_labels=None,
        decomposition_algorithm="isomap",
        clustering_algorithm="kmeans",
        planes: Sequence = ("u", "v", "y"),
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

        self.decomposition = None
        self.clustering_labels = None
        self.clustering_score = None

        self.out_path = out_path
        self.plot_name = name
        self.title = title

    def decompose(self, clustering_decomp: bool = True):
        if clustering_decomp:
            n_components = self.n_clustering_components
        else:
            n_components = self.n_components
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

        fit = BatchedFit(
            self.data_loader,
            decomp,
            max_features=30,
            planes=self.planes,
            embedding_function=self.embedding_function,
        )
        fit.fit()
        decomposition = fit.transform()
        return decomposition

    def cluster(self):
        try:
            clustering_algo = {
                "kmeans": KMeans(n_clusters=self.n_clusters).fit_transform,
                "dbscan": DBSCAN().fit_predict,
            }[self.clustering_algorithm]
        except KeyError:
            raise NotImplementedError("Clustering algorithm not implemented")

        if self.decomposition is None:
            self.decomposition = self.decompose(clustering_decomp=True)

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
            self.decomposition = self.decompose(clustering_decomp=True)

        if self.clustering_labels is None:
            self.cluster()

        if self.n_clustering_components != self.n_components:
            self.decomposition = self.decompose(clustering_decomp=False)

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

    def __call__(self):
        self.visualize()
