#!/bin/bash
#SBATCH --job-name=cot-vllm-bench
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
# Default: open HF model (no Meta license). For gated models (e.g. meta-llama/*),
# set HF_TOKEN before sbatch (or huggingface-cli login), or put the token in
# .hf_token in the repo root (gitignored). vLLM pulls from HF unless --model is a local path.
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATASET="${DATASET:-dataset.json}"
MAX_RETRIES="${MAX_RETRIES:-5}"
NUM_PROBLEMS="${NUM_PROBLEMS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
VLLM_PORT="${VLLM_PORT:-8000}"
PYTHON="${PYTHON:-python3}"
VLLM_MAX_WAIT=10000   # seconds to wait for vllm to be ready

export OPENAI_API_KEY="$(cat ~/.openai_key)"

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
echo "Start time  : $(date)"
echo "──────────────────────────────────────────"

# ── Start vLLM server ──────────────────────────────────────────────────────────
# Client (benchmark.py / clients.py) reads these; must match the served model id.
export VLLM_MODEL
export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

# Prefer `vllm` CLI when on PATH; otherwise use the package entrypoint (typical on clusters).
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

# ── Theorem prover ────────────────────────────────────────────────────────────
export PATH="$SLURM_SUBMIT_DIR/bin:$PATH"

# ── Run benchmark ──────────────────────────────────────────────────────────────
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
    --client vllm \
    --max-retries "$MAX_RETRIES" \
    $OUTPUT_ARG \
    $NUM_PROBLEMS_ARG

echo "──────────────────────────────────────────"
echo "End time : $(date)"
