import torch
import pytorch_lightning as pl
import h5py

from nugraph.models import NuGraph2
from nugraph.data import H5DataModule, H5Dataset
from nugraph import util

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from pynuml.io import H5Interface


class Load:
    def __init__(
        self,
        checkpoint_path="./paper.ckpt",
        data_path="/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023.gnn.h5",
        message_passing_steps=5,
        batch_size=16,
        test=False,
        planes=["u", "v", "y"],
        n_batches=None,
        prune_index=None,
        load_data=True,
        add_features=False,
        load_model=True,
        auto_load=True,
    ) -> None:
        self.message_passing_steps = message_passing_steps
        self.data_path = data_path
        self.test = test
        self.planes = planes
        self.add_features = add_features
        if self.test:
            n_batches = 1

        if load_data and auto_load:
            self.data = self.load_data(
                data_path, batch_size=batch_size, n_batches=n_batches
            )

        if load_model and auto_load:
            try:
                self.model = self.load_model(checkpoint_path, prune_index=prune_index)

            except Exception as e:
                print(e)
                print("Could not load checkpoint, using an untrained network")
                self.model = NuGraph2()

    def load_model(self, checkpoint_path, graph=NuGraph2, prune_index=None):
        # Assumed pre-trained model that can perform inference on the loaded data

        try:
            model = graph.load_from_checkpoint(
                checkpoint_path,
                num_iters=self.message_passing_steps,
                event_head=False,
                semantic_head=True,
                filter_head=True,
                planar_features=64,
                nexus_features=16,
                vertex_features=40,
                prune_feature_index=prune_index,
            )

        except RuntimeError:
            model = graph.load_from_checkpoint(
                checkpoint_path,
                num_iters=self.message_passing_steps,
                planar_features=64,
                nexus_features=16,
                vertex_features=40,
                event_head=False,
                semantic_head=True,
                filter_head=True,
                map_location=torch.device("cpu"),
                prune_feature_index=prune_index,
            )
        model.eval()
        return model

    def single_graphs(self, dataset, graph_index):
        nodes = {}
        for plane in self.planes:
            nodes[plane] = dataset[plane]["batch"] == graph_index
        graph = dataset.subgraph(nodes)
        return graph

    def load_data(self, data_path, batch_size=16, n_batches=None):
        try:
            data = H5DataModule(
                data_path, batch_size=batch_size, add_features=self.add_features
            ).val_dataloader()
        except Exception as e:
            print(f"WARNING: {e}, loading 'test' samples.")
            data = DataLoader(
                H5Dataset(data_path, samples=["test"]), batch_size=batch_size
            )

        if n_batches is not None:
            data = DataLoader(data.dataset[:n_batches], batch_size=batch_size)

        try:
            print("INFO: Batching Data")
            events = data.dataset[0]["sp"]["batch"].unique()
            batches = [self.single_graphs(data.dataset[0], index) for index in events]
            return batches

        except (IndexError, KeyError):
            print("INFO: returning dataset as dataset object.")
            return data.dataset

    def load_test_data(self, data_path, batch_size=1):
        with h5py.File(data_path, "a") as f:
            data = list(f["dataset"])
            if "samples/train" not in f:
                f["samples/train"] = data
                f["samples/validation"] = data
                f["samples/test"] = data
            if "datasize/train" not in f:
                f["datasize/train"] = [0 for _ in range(len(data))]

            f.close()
        try:
            dataset = H5DataModule(data_path, batch_size).test_dataloader()
        except Exception:
            H5DataModule.generate_norm(data_path, batch_size)
            dataset = H5DataModule(data_path, batch_size).test_dataloader()

        return dataset

    @staticmethod
    def unpack(data_batch, planes=["u", "v", "y"]):
        try:
            data_batch = Batch.from_data_list([datum for datum in data_batch])
        except Exception:
            # Isn't an iterable
            pass

        return (
            data_batch.collect("x"),
            {p: data_batch[p, "plane", p].edge_index for p in planes},
            {p: data_batch[p, "nexus", "sp"].edge_index for p in planes},
            torch.empty(data_batch["sp"].num_nodes, 0),
            {
                p: data_batch[p].get(
                    "batch", torch.empty(data_batch["sp"].num_nodes, 0)
                )
                for p in planes
            },
        )

    def make_predictions(self):
        accelerator, device = util.configure_device()
        trainer = pl.Trainer(accelerator=accelerator, logger=False, devices=device)
        predictions = trainer.predict(self.model, dataloaders=self.data)
        return predictions

    def save_mini_batch(self):
        batch = self.data.dataset
        h5_file = h5py.File(name="./test_data.h5", mode="w")
        interface = H5Interface(h5_file)
        interface.save("validation", batch)

        with h5_file as f:
            f["planes"] = self.planes
            f["semantic_classes"] = ["MIP", "HIP", "shower", "michel", "diffuse"]

        f.close()
