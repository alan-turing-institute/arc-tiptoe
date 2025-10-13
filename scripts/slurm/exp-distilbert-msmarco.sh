#!/bin/bash
#SBATCH --account vjgo8416-co-beagle
#SBATCH --qos turing
#SBATCH --job-name exp-msmarco-distilbert
#SBATCH --time 0-5:0:0
#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --cpus-per-gpu 36
#SBATCH --output /bask/projects/v/vjgo8416-co-beagle/slurm_logs/exp-distilbert-msmarco_%j.out
#SBATCH --array 0-10

# Load required modules here
module purge
module load baskerville
module load bask-apps/live
module load GCC/11.3.0

# Activate venv
source /bask/projects/v/vjgo8416-co-beagle/arc-tiptoe/.venv/bin/activate

# Set environment variables here
export IR_DATASETS_TMP=/bask/projects/v/vjgo8416-co-beagle/tmp_ir_datasets
export IR_DATASETS_HOME=/bask/projects/v/vjgo8416-co-beagle/ir_datasets_cache
export HUGGINGFACE_HUB_CACHE=/bask/projects/v/vjgo8416-co-beagle/.cache
export CGO_CXXFLAGS="-I/bask/projects/v/vjgo8416-co-beagle/software/include/SEAL-4.1 -I/bask/projects/v/vjgo8416-co-beagle/software/SEAL/native/src -I/bask/projects/v/vjgo8416-co-beagle/software/SEAL/build"
export CGO_LDFLAGS="-L/bask/projects/v//vjgo8416-co-beagle/software/lib64 -lseal-4.1" 
export PATH=$PATH:/bask/projects/v/vjgo8416-co-beagle/software/go/go/bin

# Set script variables here
export SEARCH_CONFIG_PATH=/bask/projects/v/vjgo8416-co-beagle/arc-tiptoe/configs/distilbert_config_post_embeddings_search_config.json
export QUERIES_PATH=/bask/projects/v/vjgo8416-co-beagle/arc-tiptoe/processed_queries/msmarco/msmarco-document_dev/msmarco-distilbert-base-tas-b_shards/shard_${SLURM_ARRAY_TASK_ID}.csv
export SAVE_PATH=/bask/projects/v/vjgo8416-co-beagle/arc-tiptoe/search_results/msmarco/msmarco-document_dev/distilbert/result_${SLURM_ARRAY_TASK_ID}.csv
export SEARCH_PATH=/bask/projects/v/vjgo8416-co-beagle/arc-tiptoe/search


python3 scripts/run_search_experiment.py --json_search_config_path ${SEARCH_CONFIG_PATH} --queries_path ${QUERIES_PATH} --save_path ${SAVE_PATH} --search_dir ${SEARCH_PATH}