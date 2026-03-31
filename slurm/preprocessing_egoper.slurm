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

echo "PREPROCESSING egoper START ($(date))"
echo "Host: $(hostname)"

source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate protas_new

if [ -f "egoper_preprocessing.py" ]; then
    ipython egoper_preprocessing.py
fi

echo "PREPROCESSING GTEA COMPLETE! ($(date))"
echo "Disk usage: $(du -sh data/ | cut -f1)"