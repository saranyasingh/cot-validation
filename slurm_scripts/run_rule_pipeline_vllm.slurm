#!/bin/bash
#SBATCH --job-name=cot-rule-vllm
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                 # request 1 GPU; change to gpu:a100:1 etc. if needed
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner              # change to your cluster's GPU partition name

# ── Optional: email notifications ──────────────────────────────────────────────
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=saranyasingh@college.harvard.edu

# ── Config ─────────────────────────────────────────────────────────────────────
# Pass VLLM_MODEL at submission time:
#   sbatch --export=ALL,VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct run_rule_pipeline_vllm.slurm
# Or set it directly below.
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATASET="${DATASET:-proofwriter.json}"
MAX_RETRIES="${MAX_RETRIES:-3}"
LIMIT="${LIMIT:-}"              # number of items to process; empty = all
OUTPUT_DIR="${OUTPUT_DIR:-}"    # leave empty to use default proofwriter_outputs/
VLLM_PORT="${VLLM_PORT:-8000}"
PYTHON="${PYTHON:-python3}"
VLLM_MAX_WAIT=600   # seconds to wait for vllm to be ready

export OPENAI_API_KEY="$(cat ~/.openai_key 2>/dev/null || echo '')"

# ── Environment setup ──────────────────────────────────────────────────────────
# Slurm runs a non-interactive shell, so ~/.bashrc conda init doesn't run.
# Source conda directly, then activate.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "ERROR: Could not find conda installation." && exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate vllm-inference || { echo "ERROR: Failed to activate conda env 'vllm-inference'"; exit 1; }

mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

# Load HF token for gated models (e.g. meta-llama/*)
if [ -z "${HF_TOKEN:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/.hf_token" ]; then
    export HF_TOKEN="$(tr -d ' \n\r\t' < "${SLURM_SUBMIT_DIR}/.hf_token")"
fi

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU(s)      : $CUDA_VISIBLE_DEVICES"
echo "Python      : $PYTHON ($(command -v "$PYTHON" 2>/dev/null || echo not found))"
echo "Model       : $VLLM_MODEL"
echo "Dataset     : $DATASET"
echo "Max retries : $MAX_RETRIES"
echo "Limit       : ${LIMIT:-all}"
echo "Start time  : $(date)"
echo "──────────────────────────────────────────"

# ── Start vLLM server ──────────────────────────────────────────────────────────
export VLLM_MODEL
export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

if command -v vllm >/dev/null 2>&1; then
    echo "Starting vLLM via: vllm serve ..."
    vllm serve "$VLLM_MODEL" \
        --port "$VLLM_PORT" \
        --dtype auto \
        --max-model-len 4096 \
        &
else
    echo "Starting vLLM via: $PYTHON -m vllm.entrypoints.openai.api_server ..."
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --port "$VLLM_PORT" \
        --dtype auto \
        --max-model-len 4096 \
        &
fi
VLLM_PID=$!

# Ensure vLLM is killed when the job exits for any reason
trap "echo 'Shutting down vLLM (PID $VLLM_PID)...'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

# ── Wait for vLLM to be ready ──────────────────────────────────────────────────
echo "Waiting for vLLM server on port $VLLM_PORT..."
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $VLLM_MAX_WAIT ]; then
        echo "ERROR: vLLM did not start within ${VLLM_MAX_WAIT}s. Exiting."
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "vLLM is ready after ${ELAPSED}s."

# ── Theorem prover on PATH ────────────────────────────────────────────────────
export PATH="$SLURM_SUBMIT_DIR/bin:$PATH"

# ── Build optional flags ───────────────────────────────────────────────────────
OUTPUT_ARG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_ARG="--output-dir $OUTPUT_DIR"
fi
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# ── Run rule pipeline ──────────────────────────────────────────────────────────
"$PYTHON" rule_pipeline.py \
    --dataset-file "$DATASET" \
    --client vllm \
    --model "$VLLM_MODEL" \
    --max-retries "$MAX_RETRIES" \
    $OUTPUT_ARG \
    $LIMIT_ARG

echo "──────────────────────────────────────────"
echo "End time : $(date)"
