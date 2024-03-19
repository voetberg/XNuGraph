import h5py
import matplotlib.pyplot as plt
import numpy as np
import json
from nugraph.data import H5Dataset
import torch

from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals

file_name = "./analysis_subset.h5"


def precision_recall():
    fig, subplots = plt.subplots(2, 3, sharex=True, figsize=(12, 8))
    planes = ["u", "v", "y"]
    dataset = {"precision": {}, "recall": {}}
    with h5py.File(file_name, "r") as f:
        classes = list(f["semantic_classes"])
        for plane_index, plane in enumerate(planes):
            precision = np.array(f["precision"][plane])
            recall = np.array(f["recall"][plane])
            dataset["precision"][plane] = {
                class_index: precision[:, class_index].tolist()
                for class_index in range(0, 5)
            }
            dataset["recall"][plane] = {
                class_index: recall[:, class_index].tolist()
                for class_index in range(0, 5)
            }

            subplots[0, plane_index].hist(precision)
            subplots[1, plane_index].hist(recall)

            subplots[0, plane_index].set_title(plane)

        subplots[0, 0].set_ylabel("Precision")
        subplots[1, 0].set_ylabel("Recall")

    subplots[0, 0].set_xlim([0.25, 1.05])

    fig.tight_layout()
    fig.legend(classes)
    plt.savefig("results/scrap/precison_recall.png")

    json.dump(dataset, open("results/scrap/test_dataset_metrics.json", "w"))


def test_plot_distribution(graph_index=1):
    fig, subplots = plt.subplots(1, 3, sharex=True, figsize=(12, 4))
    planes = ["u", "v", "y"]
    graphs = H5Dataset(file_name, ["test"]).get(0)

    batches = graphs.collect("batch")
    nodes = {}
    for batch in batches:
        nodes[batch] = batches[batch] == graph_index
    single_graph_sample = graphs.subgraph(nodes)

    y_semantic = single_graph_sample.collect("y_semantic")
    x_semantic = single_graph_sample.collect("x_semantic")

    for plane_index, plane in enumerate(planes):
        y_hat = torch.argmax(torch.softmax(x_semantic[plane], axis=1), axis=1)

        subplots[plane_index].hist(y_hat, label="y_hat", alpha=0.6)
        subplots[plane_index].hist(y_semantic[plane], label="y", alpha=0.6)

    plt.legend()
    plt.savefig(f"results/scrap/index_{graph_index}_labels.png")
    plt.close("all")

    EdgeVisuals().plot(
        single_graph_sample,
        class_plot=True,
        title=f"index_{graph_index}",
        outdir="results/scrap/",
    )


if __name__ == "__main__":
    test_plot_distribution()
