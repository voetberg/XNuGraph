#!/bin/bash
set -e 

checkpoint=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt
algorithm=GNNExplainEdges
outfile=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/XNuGraph/edge_explain
data_path=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/XNuGraph/analysis_subset.h5
n_batches=10
batch_size=16
conda_env=/work1/fwk/maggiev/miniforge3/bin
env_name=xnugraph

source $conda_env/activate 
conda activate $env_name
python3 /work1/fwk/maggiev/NuGraph/scripts/explain.py --checkpoint $checkpoint  --algorithm $algorithm  --outfile $outfile --data_path $data_path