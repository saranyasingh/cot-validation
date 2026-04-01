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
VLLM_MODEL="${VLLM_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"  # any HF model
DATASET="${DATASET:-dataset.json}"
MAX_RETRIES="${MAX_RETRIES:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_WAIT=300   # seconds to wait for vllm to be ready

# ── Environment setup ──────────────────────────────────────────────────────────
# source ~/.bashrc
# conda activate my-env
# OR: source /path/to/venv/bin/activate

mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU(s)      : $CUDA_VISIBLE_DEVICES"
echo "Model       : $VLLM_MODEL"
echo "Dataset     : $DATASET"
echo "Max retries : $MAX_RETRIES"
echo "Start time  : $(date)"
echo "──────────────────────────────────────────"

# ── Start vLLM server ──────────────────────────────────────────────────────────
export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

vllm serve "$VLLM_MODEL" \
    --port "$VLLM_PORT" \
    --dtype auto \
    --max-model-len 4096 \
    &
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

# ── Run benchmark ──────────────────────────────────────────────────────────────
OUTPUT_ARG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_ARG="--output-dir $OUTPUT_DIR"
fi

python benchmark.py \
    --dataset "$DATASET" \
    --client vllm \
    --max-retries "$MAX_RETRIES" \
    $OUTPUT_ARG

echo "──────────────────────────────────────────"
echo "End time : $(date)"
