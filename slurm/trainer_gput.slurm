#!/bin/bash
#SBATCH --job-name=Protas_Train_asformer
#SBATCH --partition=allgroups
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --gres=mps:30              # shared mode, 30% GPU (~2.4GB su RTX 2000 Ada)
#SBATCH --error=slurm/logs/Protas_Train_asformer.err
#SBATCH --output=slurm/logs/Protas_Train_asformer.out



source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate gpuenv


python main.py \
  --action train \
  --dataset EgoPER_action_segmentation\
  --split 1 \
  --exp_id egoper-asformer-causal-graph \
  --causal \
  --graph \
  --learnable_graph \
  --backbone asformer \
  --num_epochs 70 \
  --batch_size 4 &

TRAIN_PID=$!


nvidia-smi --query-compute-apps=pid,used_gpu_memory \
           --format=csv,noheader --loop=5 \
           | grep "^$TRAIN_PID" > slurm/logs/gpu_log.csv &

GPU_PID=$!

wait $TRAIN_PID 
kill $GPU_PID


echo "=== PEAK GPU MEMORY ==="
awk -F',' 'NR>1 {gsub(/ MiB/,"",$2); if($2+0 > max) max=$2+0} END {print max " MiB"}' slurm/logs/gpu_log.csv