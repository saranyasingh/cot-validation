#!/bin/bash
#SBATCH --job-name=cot-benchmark
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=sapphire        # change to your cluster's partition name

# ── Optional: email notifications ──────────────────────────────────────────────
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=saranyasingh@college.harvard.edu

# ── Args (override at submission time via --export or edit here) ───────────────
DATASET="${DATASET:-$SLURM_SUBMIT_DIR/../data/dataset.json}"
CLIENT="${CLIENT:-openai}"       # openai | kimi | deepseek
MAX_RETRIES="${MAX_RETRIES:-5}"
NUM_PROBLEMS="${NUM_PROBLEMS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"     # leave empty to use default benchmark_outputs/

# ── API keys — set these in your cluster environment or uncomment and fill in ──
# export OPENAI_API_KEY=""
# export KIMI_API_KEY=""
# export DEEPSEEK_API_KEY=""

# ── Environment setup ──────────────────────────────────────────────────────────
# Compute nodes often have no `python` on PATH (only python3, or after module load).
# Uncomment if your site requires it (see run_benchmark_vllm.slurm for FASRC example):
# module load python/3.12.11-fasrc02

# Create log directory
mkdir -p slurm_logs

cd "$SLURM_SUBMIT_DIR"

if [ -f "${SLURM_SUBMIT_DIR}/.venv/bin/activate" ]; then
    source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
fi
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null)}"
if [ -z "$PYTHON" ]; then
    echo "ERROR: No Python found. Load a module, create .venv, or set PYTHON=/path/to/python" >&2
    exit 1
fi

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Python      : $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown))"
echo "Dataset     : $DATASET"
echo "Client      : $CLIENT"
echo "Max retries : $MAX_RETRIES"
echo "Start time  : $(date)"
echo "──────────────────────────────────────────"

# ── Theorem prover ────────────────────────────────────────────────────────────
export PATH="$SLURM_SUBMIT_DIR/bin:$PATH"

# Build optional --output-dir flag
OUTPUT_ARG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_ARG="--output-dir $OUTPUT_DIR"
fi
NUM_PROBLEMS_ARG=""
if [ -n "$NUM_PROBLEMS" ]; then
    NUM_PROBLEMS_ARG="--num-problems $NUM_PROBLEMS"
fi

"$PYTHON" benchmark.py \
    --dataset "$DATASET" \
    --client "$CLIENT" \
    --max-retries "$MAX_RETRIES" \
    $OUTPUT_ARG \
    $NUM_PROBLEMS_ARG

echo "──────────────────────────────────────────"
echo "End time : $(date)"
