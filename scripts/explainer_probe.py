from nugraph.explain_graph.utils.load import Load
from nugraph.explain_graph.algorithms.linear_probes.probed_network import DynamicProbedNetwork
from nugraph.explain_graph.algorithms.linear_probes.probed_network import visualize_all_probes

from nugraph.data import H5DataModule, H5Dataset
import torch.multiprocessing as mp
import torch

data_path = "/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023.gnn.h5"
#data_path = "/wclustre/fwk/exatrkx/data/uboone/CHEP2023/XNuGraph/analysis_subset.h5"
checkpoint = "/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt"
outdir = "/wclustre/fwk/exatrkx/data/uboone/CHEP2023/XNuGraph/dynamic_probe/"


def main_probe(rank): 
    subset = H5DataModule(data_path=data_path, add_features=False, batch_size=1, device=rank).test_subset
    model = Load(checkpoint_path=checkpoint, data_path=data_path, load_data=False).model

    for network_target in ["encoder", "message", "decoder"]: 
        for feature_loss in ['tracks', "hipmip"]: 
            
            if network_target == "message": 
                for message_passing_step in [1, 3, 5]: 
                              
                    network = DynamicProbedNetwork(
                        model=model, 
                        data=subset, 
                        epochs=25, 
                        rank=rank, 
                        total_devices=torch.cuda.device_count(),
                        make_latent_rep=False, 
                        make_embedding_rep=False, 
                        feature_loss=feature_loss, 
                        network_target=network_target, 
                        message_passing_steps=message_passing_step, 
                        out_path=outdir
                        )
                    network.train()

            else: 
                network = DynamicProbedNetwork(
                    model=model, 
                    data=subset, 
                    epochs=40, 
                    rank=rank, 
                    total_devices=torch.cuda.device_count(),
                    make_latent_rep=False, 
                    make_embedding_rep=False, 
                    feature_loss=feature_loss, 
                    network_target=network_target, 
                    message_passing_steps=5, 
                    out_path=outdir
                    )
                network.train()

        visualize_all_probes(outdir)

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
    internal_rep = True     

    if internal_rep: 
        main_decomp()

    else: 
        world_size = torch.cuda.device_count()
        mp.spawn(main_probe, nprocs=world_size)
