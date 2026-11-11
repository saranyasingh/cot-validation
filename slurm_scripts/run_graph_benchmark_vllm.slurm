#!/bin/bash
#SBATCH --job-name=cot-graph-vllm
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # Vampire proves a graph's steps in parallel
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                     # change to gpu:a100:1 etc. if needed
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner              # change to your cluster's GPU partition name

# Graph pipeline benchmark: a local vLLM model (e.g. Qwen) writes each CoT, the graph
# agents (Claude) build an inference graph from it, and Vampire proves every step.
# Reports, for every item, whether the answer was verified and whether it was correct.
#
# Submit from inside slurm_scripts/:
#   sbatch run_graph_benchmark_vllm.slurm
#   sbatch --export=ALL,NUM_PROBLEMS=50,VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct run_graph_benchmark_vllm.slurm
#   sbatch --export=ALL,VERIFIER_CLIENT=claude,ANTHROPIC_MODEL=claude-opus-5-5 run_graph_benchmark_vllm.slurm
#   sbatch --export=ALL,DATASET=../data/math_word_problems.json,START=100,NUM_PROBLEMS=100 run_graph_benchmark_vllm.slurm

# ── Config ─────────────────────────────────────────────────────────────────────
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-$SLURM_SUBMIT_DIR/../data/dataset.json}"
NUM_PROBLEMS="${NUM_PROBLEMS:-}"          # empty = all items
START="${START:-0}"                       # index of the first item
OUTPUT_DIR="${OUTPUT_DIR:-}"              # empty = benchmark_outputs/
VERIFIER_CLIENT="${VERIFIER_CLIENT:-bedrock}"   # bedrock | claude | openai | kimi | deepseek
PROVER_TIMEOUT="${PROVER_TIMEOUT:-20}"    # Vampire seconds per step
VAMPIRE_VERSION="${VAMPIRE_VERSION:-v5.1.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
PYTHON="${PYTHON:-python3}"
VLLM_MAX_WAIT=10000   # seconds to wait for vllm to be ready

REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"

# ── API keys ───────────────────────────────────────────────────────────────────
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat ~/.openai_key 2>/dev/null || echo '')}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$(cat ~/.anthropic_key 2>/dev/null || echo '')}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-opus-5-5}"
if [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ] && [ -f ~/.bedrock_key ]; then
    export AWS_BEARER_TOKEN_BEDROCK="$(tr -d ' \n\r\t' < ~/.bedrock_key)"
fi
export AWS_REGION="${AWS_REGION:-us-east-1}"
export BEDROCK_MODEL="${BEDROCK_MODEL:-anthropic.claude-opus-5-5}"
export ANTHROPIC_MAX_TOKENS="${ANTHROPIC_MAX_TOKENS:-32000}"   # graph JSON can be long
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

# ── Environment setup ──────────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -f "/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/n/sw/Mambaforge-23.11.0-0"
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

# ── Vampire (Linux build) ──────────────────────────────────────────────────────
# tools/vampire is gitignored and may be a macOS build, so use a Linux binary in
# tools/linux/, downloading it once if needed. Set VAMPIRE_PATH to use your own.
if [ -z "${VAMPIRE_PATH:-}" ]; then
    VAMPIRE_DIR="$REPO_ROOT/tools/linux"
    if [ ! -x "$VAMPIRE_DIR/vampire" ]; then
        echo "Downloading Vampire $VAMPIRE_VERSION (Linux x64) to $VAMPIRE_DIR..."
        mkdir -p "$VAMPIRE_DIR"
        curl -fsSL -o "$VAMPIRE_DIR/vampire.zip" \
            "https://github.com/vprover/vampire/releases/download/${VAMPIRE_VERSION}/vampire-Linux-X64.zip" \
            || { echo "ERROR: Vampire download failed. Download vampire-Linux-X64.zip on a login node and unzip it into $VAMPIRE_DIR."; exit 1; }
        "$PYTHON" -m zipfile -e "$VAMPIRE_DIR/vampire.zip" "$VAMPIRE_DIR" && rm -f "$VAMPIRE_DIR/vampire.zip"
        chmod +x "$VAMPIRE_DIR/vampire"
    fi
    export VAMPIRE_PATH="$VAMPIRE_DIR/vampire"
fi
"$VAMPIRE_PATH" --version >/dev/null 2>&1 || { echo "ERROR: $VAMPIRE_PATH does not run on this node."; exit 1; }

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU(s)       : $CUDA_VISIBLE_DEVICES"
echo "Python       : $PYTHON ($(command -v "$PYTHON" 2>/dev/null || echo not found))"
echo "Reasoning    : vllm ($VLLM_MODEL)"
if [ "$VERIFIER_CLIENT" = "bedrock" ]; then
    echo "Checking     : bedrock ($BEDROCK_MODEL, $AWS_REGION)"
    echo "Bedrock key  : $([ -n "${AWS_BEARER_TOKEN_BEDROCK:-${BEDROCK_API_KEY:-}}" ] && echo set || echo MISSING)"
else
    echo "Checking     : $VERIFIER_CLIENT"
fi
echo "Dataset      : $DATASET (start $START, count ${NUM_PROBLEMS:-all})"
echo "Vampire      : $VAMPIRE_PATH ($("$VAMPIRE_PATH" --version 2>/dev/null | head -1)), ${PROVER_TIMEOUT}s per step"
echo "Start time   : $(date)"
echo "──────────────────────────────────────────"

# ── Start vLLM server ──────────────────────────────────────────────────────────
export VLLM_MODEL
export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

if command -v vllm >/dev/null 2>&1; then
    vllm serve "$VLLM_MODEL" --port "$VLLM_PORT" --dtype auto --max-model-len "$VLLM_MAX_MODEL_LEN" &
else
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" --port "$VLLM_PORT" --dtype auto --max-model-len "$VLLM_MAX_MODEL_LEN" &
fi
VLLM_PID=$!
trap "echo 'Shutting down vLLM (PID $VLLM_PID)...'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

echo "Waiting for vLLM server on port $VLLM_PORT..."
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $VLLM_MAX_WAIT ]; then
        echo "ERROR: vLLM did not start within ${VLLM_MAX_WAIT}s. Exiting."
        exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process exited before becoming ready (see .err log). Exiting."
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "vLLM is ready after ${ELAPSED}s."

# ── Run graph benchmark ────────────────────────────────────────────────────────
ARGS=(--dataset "$DATASET" --client vllm --verifier "$VERIFIER_CLIENT"
      --start "$START" --prover-timeout "$PROVER_TIMEOUT")
[ -n "$NUM_PROBLEMS" ] && ARGS+=(--num-problems "$NUM_PROBLEMS")
[ -n "$OUTPUT_DIR" ] && ARGS+=(--output-dir "$OUTPUT_DIR")

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
"$PYTHON" -m graph_pipeline.benchmark "${ARGS[@]}"

echo "──────────────────────────────────────────"
echo "End time : $(date)"
