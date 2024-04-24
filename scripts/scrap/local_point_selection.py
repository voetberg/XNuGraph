from nugraph.explain_graph.utils.load import Load
from nugraph.explain_graph.utils.edge_visuals import EdgeVisuals
import json
from pynuml.io import H5Interface
from torch_geometric.data import Batch
import h5py
import torch
import numpy as np
import os

planes = ["u", "v", "y"]


def hip_criteron(graph_prediction, graph_true):
    num_possible = {plane: len((graph_true[plane] == 1).nonzero()) for plane in planes}
    if sum([num_possible[plane] for plane in planes]) == 0:
        return {
            "num_correct": 0,
            "num_incorrect": 0,
            "correct_hits": {plane: [] for plane in planes},
            "incorrect_hits": {plane: [] for plane in planes},
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
    return {
        "num_correct": np.mean(
            [len(correct_index[plane].unique()) for plane in planes]
        ),
        "num_incorrect": np.mean([len(mip_index[plane].unique()) for plane in planes]),
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
            "num_correct": 0,
            "num_incorrect": 0,
            "correct_hits": {plane: [] for plane in planes},
            "incorrect_hits": {plane: [] for plane in planes},
        }

    is_correct = {
        plane: torch.logical_and(graph_prediction[plane] == 4, graph_true[plane] == 4)
        for plane in planes
    }
    is_incorrect = {
        # plane: torch.logical_or(
        #     torch.logical_and(graph_prediction[plane] == 4, graph_true[plane] != 4),
        #     torch.logical_and(graph_prediction[plane] != 4, graph_true[plane] == 4),
        # )
        plane: torch.logical_and(graph_prediction[plane] != 4, graph_true[plane] == 4)
        for plane in planes
    }
    correct_index = {
        plane: is_correct[plane].nonzero(as_tuple=True)[0] for plane in planes
    }
    incorrect_index = {
        plane: is_incorrect[plane].nonzero(as_tuple=True)[0] for plane in planes
    }
    return {
        "num_correct": np.mean(
            [len(correct_index[plane].unique()) for plane in planes]
        ),
        "num_incorrect": np.mean(
            [len(incorrect_index[plane].unique()) for plane in planes]
        ),
        "correct_hits": {
            plane: correct_index[plane].numpy().tolist() for plane in planes
        },
        "incorrect_hits": {
            plane: incorrect_index[plane].numpy().tolist() for plane in planes
        },
    }


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

def save(data, metrics, node_indices, outfile, name_prefix=""):

    keep_dict = {int(key): metrics[key] for key in node_indices}

    selected_data = [data[index] for index in node_indices]
    [
        EdgeVisuals().event_plot(d, outdir=out_path, file_name=f"event_{n}.png")
        for d, n in zip(selected_data, node_indices)
    ]

    subset = Batch.from_data_list([item for item in selected_data])
    h5_file = h5py.File(name=f"{outfile}/{name_prefix}results.h5", mode="w")
    interface = H5Interface(h5_file)
    interface.save("test", subset)

    with open(f"{outfile}/{name_prefix}results.json", "w") as f:
        json.dump(keep_dict, f)


if __name__ == "__main__":

    checkpoint = '/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt'
    data_path = '/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023.gnn.h5'
    #data_path = "./test_data.h5"
    #out_path = "./test_data/local_hits"
    out_path = '/wclustre/fwk/exatrkx/data/uboone/CHEP2023/XNuGraph/analysis_subset_local_hits'

    os.makedirs(out_path, exist_ok=True)

    load = Load(checkpoint_path=checkpoint, data_path=data_path)
    data = load.make_predictions()

    results = []
    for index, graph in enumerate(data):
        results.append(evaluate_graph(graph, criteons=[hip_criteron, michel_criteon]))

    n_events = 8
    michel_events = np.argpartition([result["michel_criteon"]["num_incorrect"] for result in results], -1*n_events)[-1*n_events:].tolist()
    hip_events = np.argpartition([result["hip_criteron"]["num_incorrect"] for result in results], -1*n_events)[-1*n_events:].tolist()

    save(data, results, michel_events, outfile=out_path, name_prefix="michel_")
    save(data, results, hip_events, outfile=out_path, name_prefix="hip_")