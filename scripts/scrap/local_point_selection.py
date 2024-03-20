from nugraph.explain_graph.utils.load import Load
import json
from pynuml.io import H5Interface
from torch_geometric.data import Batch
import h5py
import torch
import numpy as np

planes = ["u", "v", "y"]


def hip_criteron(graph_prediction, graph_true):
    num_possible = {plane: len((graph_true[plane] == 1).nonzero()) for plane in planes}
    if sum([num_possible[plane] for plane in planes]) == 0:
        return {
            "per_correct": np.NaN,
            "per_incorrect": np.NaN,
            "correct_hits": {plane: [] for plane in planes},
            "incorrect_hits": {plane: [] for plane in planes},
        }
    num_possible = {
        plane: num_possible[plane] if num_possible[plane] != 0 else 1
        for plane in planes
    }

    is_correct = {
        plane: torch.logical_and(graph_prediction[plane] == 1, graph_true[plane] == 1)
        for plane in planes
    }
    is_labeled_mip = {
        plane: torch.logical_and(graph_prediction[plane] == 0, graph_true[plane] == 1)
        for plane in planes
    }
    correct_index = {
        plane: is_correct[plane].nonzero(as_tuple=True)[0] for plane in planes
    }
    mip_index = {
        plane: is_labeled_mip[plane].nonzero(as_tuple=True)[0] for plane in planes
    }
    total_possible = sum([num_possible[plane] for plane in planes])
    return {
        "per_correct": len(
            torch.concat([correct_index[plane] for plane in planes]).unique()
        )
        / total_possible,
        "per_incorrect": len(
            torch.concat([mip_index[plane] for plane in planes]).unique()
        )
        / total_possible,
        "correct_hits": {
            plane: correct_index[plane].numpy().tolist() for plane in planes
        },
        "incorrect_hits": {
            plane: mip_index[plane].numpy().tolist() for plane in planes
        },
    }


def michel_criteon(graph_prediction, graph_true):
    num_possible = {plane: len((graph_true[plane] == 4).nonzero()) for plane in planes}
    if sum([num_possible[plane] for plane in planes]) == 0:
        return {
            "per_correct": np.NaN,
            "per_incorrect": np.NaN,
            "correct_hits": {plane: [] for plane in planes},
            "incorrect_hits": {plane: [] for plane in planes},
        }
    num_possible = {
        plane: num_possible[plane] if num_possible[plane] != 0 else 1
        for plane in planes
    }
    is_correct = {
        plane: torch.logical_and(graph_prediction[plane] == 4, graph_true[plane] == 4)
        for plane in planes
    }
    is_incorrect = {
        plane: torch.logical_or(
            torch.logical_and(graph_prediction[plane] == 4, graph_true[plane] != 4),
            torch.logical_and(graph_prediction[plane] != 4, graph_true[plane] == 4),
        )
        for plane in planes
    }
    correct_index = {
        plane: is_correct[plane].nonzero(as_tuple=True)[0] for plane in planes
    }
    incorrect_index = {
        plane: is_incorrect[plane].nonzero(as_tuple=True)[0] for plane in planes
    }
    total_possible = sum([num_possible[plane] for plane in planes])
    return {
        "per_correct": len(
            torch.concat([correct_index[plane] for plane in planes]).unique()
        )
        / total_possible,
        "per_incorrect": len(
            torch.concat([incorrect_index[plane] for plane in planes]).unique()
        )
        / total_possible,
        "correct_hits": {
            plane: correct_index[plane].numpy().tolist() for plane in planes
        },
        "incorrect_hits": {
            plane: incorrect_index[plane].numpy().tolist() for plane in planes
        },
    }


def save(selected_data, node_indices, outfile):
    subset = Batch.from_data_list([item for item in selected_data])

    h5_file = h5py.File(name=f"{outfile}.h5", mode="w")
    interface = H5Interface(h5_file)
    interface.save("test", subset)

    with open(f"{outfile}.json", "w") as f:
        json.dump(node_indices, f)


def evaluate_graph(graph, criteons: list):
    true = graph.collect("y_semantic")
    predictions = {
        plane: torch.argmax(pred, dim=1)
        for plane, pred in graph.collect("x_semantic").items()
    }
    results = {}
    for criteon in criteons:
        results[criteon.__name__] = criteon(predictions, true)

    return results


if __name__ == "__main__":
    checkpoint = "./paper.ckpt"
    data_path = "./analysis_subset.h5"
    out_path = "./hip_mip_michel_local_revision"

    load = Load(checkpoint_path=checkpoint, data_path=data_path)

    results = []
    for index, graph in enumerate(load.data):
        results.append(evaluate_graph(graph, criteons=[hip_criteron, michel_criteon]))

    keep_indices = [
        np.nanargmax([result["hip_criteron"]["per_correct"] for result in results]),
        np.nanargmax([result["hip_criteron"]["per_incorrect"] for result in results]),
        np.nanargmax([result["michel_criteon"]["per_correct"] for result in results]),
        np.nanargmax([result["michel_criteon"]["per_incorrect"] for result in results]),
        np.nanargmin([result["michel_criteon"]["per_correct"] for result in results]),
        np.nanargmin([result["hip_criteron"]["per_correct"] for result in results]),
        np.nanargmin([result["michel_criteon"]["per_incorrect"] for result in results]),
        np.nanargmin([result["hip_criteron"]["per_incorrect"] for result in results]),
    ]
    print(keep_indices)

    keep_dict = {int(key): results[key] for key in keep_indices}
    selected_data = [load.data[index] for index in keep_indices]

    save(selected_data, keep_dict, out_path)
