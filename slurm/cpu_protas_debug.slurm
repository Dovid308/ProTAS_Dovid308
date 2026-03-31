#!/bin/bash
# ProTAS DEBUG - UniPD Math HPC (partition=debug)

#SBATCH --job-name=ProTAS_debug
#SBATCH --partition=debug
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --error=slurm/logs/protas_debug.err
#SBATCH --output=slurm/logs/protas_debug.out

echo "ProTAS DEBUG START ($(date))"
echo "Host: $(hostname)"
echo "Partition: debug | Account: thesis"
echo "CPUs: 4 | Mem: 4G | Time: 20min"
echo "Working dir: $(pwd)"


source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate protas_new  # o il tuo env
echo "Conda: $(conda info --envs | grep '*' ) | Python: $(python --version)"


python main.py \
  --action train \
  --dataset gtea \
  --split 1 \
  --exp_id gtae-mstcn-causal-graph \
  --causal \
  --graph \
  --learnable_graph \
  --backbone mstcn \
  --num_epochs 10 \
  --batch_size 4

echo "SUMMARY:"
ls -la models/protas_debug/gtea/split_1/ 2>/dev/null || echo "No models saved"
echo "Disk usage: $(du -sh . | cut -f1)"
echo "End: $(date)"
echo "DEBUG COMPLETE!"
