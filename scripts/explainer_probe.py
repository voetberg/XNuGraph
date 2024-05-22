from nugraph.explain_graph.utils.load import Load
from nugraph.explain_graph.algorithms.linear_probes.probed_network import DynamicProbedNetwork
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

    for network_target in ['decoder']: # [ "encoder", "message", "decoder"]: 
        for feature_loss in ['tracks', "hipmip"]: 

            network = DynamicProbedNetwork(
                model=model, data=subset, epochs=15,  rank=rank, total_devices=2, make_latent_rep=False, make_embedding_rep=False, feature_loss=feature_loss, network_target=network_target
                )
            network.train()

def main_decomp(): 
    subset = H5DataModule(data_path=data_path, add_features=False, batch_size=1).test_subset
    model = Load(checkpoint_path=checkpoint, data_path=data_path, load_data=False).model

    for network_target in [ "encoder", "message", "decoder"]: 
        for feature_loss in ['tracks', "hipmip"]: 

            network = DynamicProbedNetwork(
                model=model, data=subset, epochs=15,  make_latent_rep=True, make_embedding_rep=False, feature_loss=feature_loss, network_target=network_target
                )
            network.make_embedding_rep()


if __name__ == "__main__": 
    internal_rep = False     

    if internal_rep: 
        main_decomp()

    else: 
            
        world_size = torch.cuda.device_count()
        mp.spawn(main_probe, nprocs=world_size)
