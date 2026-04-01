#!/bin/bash
#SBATCH --job-name=cot-benchmark
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu          # change to your cluster's partition name

# ── Optional: email notifications ──────────────────────────────────────────────
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=saranyasingh@college.harvard.edu

# ── Args (override at submission time via --export or edit here) ───────────────
DATASET="${DATASET:-dataset.json}"
CLIENT="${CLIENT:-openai}"       # openai | kimi | deepseek
MAX_RETRIES="${MAX_RETRIES:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-}"     # leave empty to use default benchmark_outputs/

# ── API keys — set these in your cluster environment or uncomment and fill in ──
# export OPENAI_API_KEY=""
# export KIMI_API_KEY=""
# export DEEPSEEK_API_KEY=""

# ── Environment setup ──────────────────────────────────────────────────────────
# Activate your conda/venv environment if needed:
# source ~/.bashrc
# conda activate my-env
# OR:
# source /path/to/venv/bin/activate

# Create log directory
mkdir -p slurm_logs

cd "$SLURM_SUBMIT_DIR"

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Dataset     : $DATASET"
echo "Client      : $CLIENT"
echo "Max retries : $MAX_RETRIES"
echo "Start time  : $(date)"
echo "──────────────────────────────────────────"

# Build optional --output-dir flag
OUTPUT_ARG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_ARG="--output-dir $OUTPUT_DIR"
fi

python benchmark.py \
    --dataset "$DATASET" \
    --client "$CLIENT" \
    --max-retries "$MAX_RETRIES" \
    $OUTPUT_ARG

echo "──────────────────────────────────────────"
echo "End time : $(date)"
