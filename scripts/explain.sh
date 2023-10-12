checkpoint=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/paper.ckpt 
algorithm=GNNExplainer
outfile=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/test.gnn.h5
data_path=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/CHEP2023-small.evt.h5
batch_size=16
test=True

srun --unbuffered --pty -A fwk --partition=gpu_gce --gres=gpu:v100:1 --qos=test --nodes=1 --time=01:00:00 --ntasks-per-node=4 python3 scripts/explain.py --checkpoint $checkpoint  --algorithm $algorithm  --outfile $outfile --data_path $data_path --batch_size $batch_size --test $test
