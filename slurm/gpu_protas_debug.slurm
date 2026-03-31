#!/bin/bash
#SBATCH --job-name=ProTAS_gtea
#SBATCH --partition=debug
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=mps:30              # shared mode, 30% GPU (~2.4GB su RTX 2000 Ada)
#SBATCH --error=slurm/logs/protas.err
#SBATCH --output=slurm/logs/protas.out

# --- GPU monitoring in background ---
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
           --format=csv --loop=5 > slurm/logs/gpu_log.csv &
GPU_PID=$!

source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate gpuenv

python main.py \
  --action train \
  --dataset gtea \
  --split 1 \
  --exp_id gtae-asformer-causal-graph \
  --causal \
  --graph \
  --learnable_graph \
  --backbone asformer \
  --num_epochs 120 \
  --batch_size 1

kill $GPU_PID

echo "=== PEAK GPU MEMORY ==="
awk -F',' 'NR>1 {gsub(/ MiB/,"",$2); if($2+0 > max) max=$2+0} END {print max " MiB"}' slurm/logs/gpu_log.csv