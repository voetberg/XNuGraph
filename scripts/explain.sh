#!/bin/bash
set -e 

checkpoint=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt 
algorithm=GNNExplainer
outfile=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/test_explain.gnn.h5
data_path=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023.gnn.h5
batch_size=16
test=True

# conda_env=/work1/fwk/maggiev/miniforge3/bin
# env_name=nugraph

# source $conda_env/activate 
# conda activate $env_name
python3 /work1/fwk/maggiev/NuGraph/scripts/explain.py --checkpoint $checkpoint  --algorithm $algorithm  --outfile $outfile --data_path $data_path --batch_size $batch_size --test $test