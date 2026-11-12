#!/bin/bash
#SBATCH --job-name=cot-formal-dspy
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # NUM_THREADS items are formalized and proved in parallel
#SBATCH --mem=16G
#SBATCH --partition=sapphire             # CPU only: CoTs come from saved runs, prompts run on Bedrock

# GEPA-optimizes the main pipeline's formal-verification prompts (FOL extraction and
# TPTP conversion) on saved benchmark CoTs. No GPU or vLLM needed.
#
# Submit from inside slurm_scripts/:
#   sbatch run_formal_optimize.slurm                                    # auto=light
#   sbatch --export=ALL,MAX_METRIC_CALLS=400 run_formal_optimize.slurm
#   sbatch --export=ALL,EVAL_ONLY=1 run_formal_optimize.slurm           # baseline test score only
#   sbatch --export=ALL,EVAL_ONLY=1,PROGRAM=../formal_dspy/runs/<run>/program.json run_formal_optimize.slurm
#
# Requires dspy in the Python environment: pip install dspy

# ── Config ─────────────────────────────────────────────────────────────────────
VERIFIER="${VERIFIER:-bedrock}"           # runs the formalization prompts
VERIFIER_MODEL="${VERIFIER_MODEL:-}"      # empty = the client's default (BEDROCK_MODEL / .env), as the pipeline uses
REFLECTION="${REFLECTION:-bedrock}"       # rewrites the prompts
REFLECTION_MODEL="${REFLECTION_MODEL:-anthropic.claude-opus-5-5}"
AUTO="${AUTO:-light}"                     # light | medium | heavy (ignored if MAX_METRIC_CALLS is set)
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-}"
NUM_THREADS="${NUM_THREADS:-8}"
SEED="${SEED:-0}"
RUNS="${RUNS:-}"                          # space-separated run dirs/globs; empty = all saved runs
EVAL_ONLY="${EVAL_ONLY:-}"
PROGRAM="${PROGRAM:-}"
VAMPIRE_VERSION="${VAMPIRE_VERSION:-v5.1.0}"

REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"

# ── API keys ───────────────────────────────────────────────────────────────────
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$(cat ~/.anthropic_key 2>/dev/null || echo '')}"
if [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ] && [ -f ~/.bedrock_key ]; then
    export AWS_BEARER_TOKEN_BEDROCK="$(tr -d ' \n\r\t' < ~/.bedrock_key)"
fi
export AWS_REGION="${AWS_REGION:-us-east-1}"
export ANTHROPIC_MAX_TOKENS="${ANTHROPIC_MAX_TOKENS:-32000}"
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

# ── Environment setup ──────────────────────────────────────────────────────────
mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

if [ -z "${PYTHON:-}" ]; then
    if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
        source "$REPO_ROOT/.venv/bin/activate"
    elif command -v conda >/dev/null 2>&1 || [ -f "/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$(conda info --base 2>/dev/null || echo /n/sw/Mambaforge-23.11.0-0)"
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV:-vllm-inference}"
    fi
    PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null)"
fi
"$PYTHON" -c "import dspy" 2>/dev/null || { echo "ERROR: dspy is not installed for $PYTHON (pip install dspy)"; exit 1; }

# ── Provers: E for the inference check, Vampire (Linux) for the arithmetic check ─
export PATH="$SLURM_SUBMIT_DIR/bin:$PATH"
if [ -z "${VAMPIRE_PATH:-}" ]; then
    VAMPIRE_DIR="$REPO_ROOT/tools/linux"
    if [ ! -x "$VAMPIRE_DIR/vampire" ]; then
        echo "Downloading Vampire $VAMPIRE_VERSION (Linux x64) to $VAMPIRE_DIR..."
        mkdir -p "$VAMPIRE_DIR"
        curl -fsSL -o "$VAMPIRE_DIR/vampire.zip" \
            "https://github.com/vprover/vampire/releases/download/${VAMPIRE_VERSION}/vampire-Linux-X64.zip" \
            && "$PYTHON" -m zipfile -e "$VAMPIRE_DIR/vampire.zip" "$VAMPIRE_DIR" \
            && rm -f "$VAMPIRE_DIR/vampire.zip" && chmod +x "$VAMPIRE_DIR/vampire" \
            || echo "WARNING: Vampire download failed; arithmetic will be checked with Python fractions."
    fi
    [ -x "$VAMPIRE_DIR/vampire" ] && export VAMPIRE_PATH="$VAMPIRE_DIR/vampire"
fi

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Python       : $PYTHON ($("$PYTHON" -c 'import dspy, sys; print(sys.version.split()[0], "dspy", dspy.__version__)'))"
echo "Verifier     : $VERIFIER (${VERIFIER_MODEL:-${BEDROCK_MODEL:-client default}})"
echo "Reflection   : $REFLECTION ($REFLECTION_MODEL)"
echo "Budget       : ${MAX_METRIC_CALLS:+$MAX_METRIC_CALLS metric calls}${MAX_METRIC_CALLS:-auto=$AUTO}"
echo "Eval only    : ${EVAL_ONLY:-no}${PROGRAM:+ (program $PROGRAM)}"
echo "E prover     : $(command -v eprover || echo MISSING)"
echo "Vampire      : ${VAMPIRE_PATH:-not found (Python fallback for arithmetic)}"
echo "Start time   : $(date)"
echo "──────────────────────────────────────────"

# ── Run ────────────────────────────────────────────────────────────────────────
ARGS=(--verifier "$VERIFIER" --reflection "$REFLECTION" --reflection-model "$REFLECTION_MODEL"
      --num-threads "$NUM_THREADS" --seed "$SEED")
[ -n "$VERIFIER_MODEL" ] && ARGS+=(--verifier-model "$VERIFIER_MODEL")
if [ -n "$EVAL_ONLY" ]; then
    ARGS+=(--eval-only)
elif [ -n "$MAX_METRIC_CALLS" ]; then
    ARGS+=(--max-metric-calls "$MAX_METRIC_CALLS")
else
    ARGS+=(--auto "$AUTO")
fi
[ -n "$PROGRAM" ] && ARGS+=(--program "$PROGRAM")
# shellcheck disable=SC2206
[ -n "$RUNS" ] && ARGS+=(--runs $RUNS)

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
"$PYTHON" -m formal_dspy.optimize "${ARGS[@]}"

echo "──────────────────────────────────────────"
echo "End time : $(date)"
