from nugraph.explain_graph.utils.load import Load
from nugraph.explain_graph.algorithms.linear_probes.probed_network import DynamicProbedNetwork

from nugraph.data import H5DataModule
import torch.multiprocessing as mp
import torch
import os 

data_path = "/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023.gnn.h5"
checkpoint = "/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt"
outdir = "/wclustre/fwk/exatrkx/data/uboone/CHEP2023/XNuGraph/dynamic_probe/"


def train_probe(model, subset, rank, loss_function, network_target, message_steps): 
    network = DynamicProbedNetwork(
        model=model, 
        data=subset, 
        epochs=25, 
        rank=rank, 
        total_devices=torch.cuda.device_count(),
        make_latent_rep=False, 
        make_embedding_rep=False, 
        feature_loss=loss_function, 
        network_target=network_target, 
        message_passing_steps=message_steps, 
        out_path=outdir)
    network.train()

def main_probe(rank): 
    subset = H5DataModule(data_path=data_path, add_features=False, batch_size=1, device=rank).test_subset
    model = Load(checkpoint_path=checkpoint, data_path=data_path, load_data=False).model

    # network = DynamicProbedNetwork(
    #     model=model, 
    #     data=subset, 
    #     epochs=1, 
    #     rank=rank, 
    #     total_devices=torch.cuda.device_count(),
    #     make_latent_rep=False, 
    #     make_embedding_rep=False, 
    #     feature_loss="wire", 
    #     network_target="encoder", 
    #     message_passing_steps=5, 
    #     out_path=f"{outdir.rstrip('/')}/test/", 
    #     test=True
    # )
    # try: 
    #     network.train()
    # finally: 
    #     for file in os.listdir(f"{outdir.rstrip('/')}/test/"): 
    #         try: 
    #             os.remove(f"{outdir.rstrip('/')}/test/{file}")
    #         except IsADirectoryError: 
    #             pass

    all_losses = [ "tracks", "hipmip", "node_slope",  "michel_conservation", 'wire', 'peak', 'integral', 'rms']
    #all_losses = []
    for feature_loss in all_losses: 
        train_probe(model, subset, rank, feature_loss, network_target="encoder", message_steps=5)

        for message_passing_step in [1, 3, 5]: 
            train_probe(model, subset, rank, feature_loss, network_target="message", message_steps=message_passing_step)


def main_decomp(): 
    subset = H5DataModule(data_path=data_path, add_features=False, batch_size=1).test_subset
    model = Load(checkpoint_path=checkpoint, data_path=data_path, load_data=False).model

    network = DynamicProbedNetwork(
        model=model, data=subset, out_path=outdir,  rank=0, 
            total_devices=1, epochs=0,  make_latent_rep=True, make_embedding_rep=True, 
            feature_loss="tracks", network_target="encoder",  message_passing_steps=5
        )
    network.network_clustering()
    
    for message_step in [1, 2, 3, 4, 5]: 
        network = DynamicProbedNetwork(
            model=model, data=subset, out_path=outdir,  rank=0, 
                total_devices=1, epochs=0,  make_latent_rep=True, make_embedding_rep=True, 
                feature_loss="tracks", network_target="message",  message_passing_steps=message_step
            )
        network.network_clustering()


if __name__ == "__main__": 
    internal_rep = False     

    if internal_rep: 
        main_decomp()

    else: 
        world_size = torch.cuda.device_count()
        mp.spawn(main_probe, nprocs=world_size)


