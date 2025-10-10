#!/bin/bash
#SBATCH --account vjgo8416-co-beagle
#SBATCH --qos turing
#SBATCH --job-name process-distilbert-msmarco
#SBATCH --time 1-0:0:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --output /bask/projects/v/vjgo8416-co-beagle/slurm_logs/pre-process-distilbert-msmarco.out

# Load required modules here
module purge
module load baskerville

# Activate venv
source /bask/projects/v/vjgo8416-co-beagle/arc-tiptoe/.venv/bin/activate

export IR_DATASETS_TMP=/bask/projects/v/vjgo8416-co-beagle/tmp_ir_datasets
export IR_DATASETS_HOME=/bask/projects/v/vjgo8416-co-beagle/ir_datasets_cache
export HUGGINGFACE_HUB_CACHE=/bask/projects/v/vjgo8416-co-beagle/.cache

python3 scripts/preprocess_from_config.py --json_config_path configs/distilbert_config_post_embeddings.json