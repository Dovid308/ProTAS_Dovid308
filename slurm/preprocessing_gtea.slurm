#!/bin/bash
#SBATCH --job-name=preprocessing_gtea
#SBATCH --partition=debug
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --error=slurm/logs/preprocessing_gtea.err
#SBATCH --output=slurm/logs/preprocessing_gtea.out

echo "PREPROCESSING GTEA START ($(date))"
echo "Host: $(hostname)"

source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate protas_new

# 1. Unzip dataset
echo "1. Unzipping dataset..."
if [ -f "data/gtea_dataset.zip" ]; then
    unzip -q -o data/gtea_dataset.zip -d data/
    mv data/content/gtea data/gtea
    rm -rf data/content/
    echo "Dataset unzipped: $(du -sh data/ | cut -f1)"
else
    echo "ERROR: gtea_dataset.zip not found!"
    exit 1
fi

# 2. Build graph/progress
echo "2. Building graph & progress..."
if [ -f "utils/write_graph.py" ]; then
    python utils/write_graph.py
fi
if [ -f "utils/write_progress_values.py" ]; then
    python utils/write_progress_values.py
fi
if [ -f "utils/groundtruth_gtea.py" ]; then
    python utils/groundtruth_gtea.py
fi

echo "PREPROCESSING GTEA COMPLETE! ($(date))"
echo "Disk usage: $(du -sh data/ | cut -f1)"