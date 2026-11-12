#!/bin/bash
#SBATCH --job-name=cot-curebench
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=sapphire        # CPU-only job; kempner_undergrads is denied on sapphire/shared,
                                    # so submit with a lab account: sbatch --account=amin_lab ...

# Runs the CoT → FOL → verify pipeline on CureBench with a CharmChat agent as the verifier,
# then prints success rates (main_pipeline/curebench_report.py).
#
# The job starts its own CharmChat backend on this node (cloned to $CHARMCHAT_DIR on first use),
# with only the MCP servers in $CHARMCHAT_SERVERS, and shuts it down at the end.
# Needs on the node: node + npm (module or PATH), uv (for the ToolUniverse server), and the
# Python env used by the other cot-validation jobs.
#
# Examples (from slurm_scripts/):
#   sbatch --account=amin_lab run_curebench_charmchat.slurm
#   sbatch --account=amin_lab --export=ALL,NUM_PROBLEMS=20,SKIP_TPTP=1 run_curebench_charmchat.slurm
#   sbatch --account=amin_lab --export=ALL,MODE=repair,MAX_RETRIES=3 run_curebench_charmchat.slurm

# ── Optional: email notifications ──────────────────────────────────────────────
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=saranyasingh@college.harvard.edu

# ── Args (override at submission time via --export or edit here) ───────────────
DATASET="${DATASET:-$SLURM_SUBMIT_DIR/../data/curebench_valset_phase1.jsonl}"
MODE="${MODE:-verify}"                  # verify (confusion matrix) | repair (full repair loop)
CLIENT="${CLIENT:-openai}"              # reasoning client: openai | claude | kimi | deepseek | bedrock
NUM_PROBLEMS="${NUM_PROBLEMS:-}"        # empty = all 459
START="${START:-0}"
QUESTION_TYPES="${QUESTION_TYPES:-}"    # e.g. "multi_choice open_ended_multi_choice"
SKIP_TPTP="${SKIP_TPTP:-}"              # verify mode: set to "1" to pass --skip-tptp
MAX_RETRIES="${MAX_RETRIES:-5}"         # repair mode
OUTPUT_DIR="${OUTPUT_DIR:-$SLURM_SUBMIT_DIR/../benchmark_outputs}"

# CharmChat verifier agent
CHARMCHAT_DIR="${CHARMCHAT_DIR:-$HOME/charmchat}"
CHARMCHAT_PROVIDER="${CHARMCHAT_PROVIDER:-anthropic}"    # anthropic | openai | bedrock (needs AWS creds)
CHARMCHAT_MODEL="${CHARMCHAT_MODEL:-claude-sonnet-4-6}"
CHARMCHAT_SERVERS="${CHARMCHAT_SERVERS:-tooluniverse}"   # comma-separated MCP servers the agent may use
CHARMCHAT_PORT="${CHARMCHAT_PORT:-$((3100 + ${SLURM_JOB_ID:-0} % 800))}"
CHARMCHAT_READY_WAIT="${CHARMCHAT_READY_WAIT:-900}"      # seconds to wait for the MCP servers to connect
NODE_MODULE="${NODE_MODULE:-nodejs}"                     # module to load if node is not on PATH

# ── API keys ───────────────────────────────────────────────────────────────────
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat ~/.openai_key 2>/dev/null || echo '')}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$(cat ~/.anthropic_key 2>/dev/null || echo '')}"
if [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ] && [ -f ~/.bedrock_key ]; then
    export AWS_BEARER_TOKEN_BEDROCK="$(tr -d ' \n\r\t' < ~/.bedrock_key)"
fi
export AWS_REGION="${AWS_REGION:-us-east-1}"
export BEDROCK_MODEL="${BEDROCK_MODEL:-anthropic.claude-sonnet-5-5}"
if [ -f "${SLURM_SUBMIT_DIR}/../.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${SLURM_SUBMIT_DIR}/../.env"
    set +a
fi
# Optional: FDA_API_KEY / SEMANTIC_SCHOLAR_API_KEY raise ToolUniverse rate limits (read from env)

mkdir -p slurm_logs
cd "$SLURM_SUBMIT_DIR"

# ── Python ─────────────────────────────────────────────────────────────────────
if [ -f "${SLURM_SUBMIT_DIR}/../.venv/bin/activate" ]; then
    source "${SLURM_SUBMIT_DIR}/../.venv/bin/activate"
fi
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null)}"
if [ -z "$PYTHON" ]; then
    echo "ERROR: No Python found. Load a module, create .venv, or set PYTHON=/path/to/python" >&2
    exit 1
fi

# ── Node and uv (for CharmChat) ────────────────────────────────────────────────
if ! command -v node >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load "$NODE_MODULE" 2>/dev/null || true
fi
if ! command -v node >/dev/null 2>&1; then
    echo "ERROR: node not found. Load a Node.js module (NODE_MODULE=...) or install it, e.g. via nvm." >&2
    exit 1
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if [[ ",$CHARMCHAT_SERVERS," == *",tooluniverse,"* ]] && ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found (needed for the ToolUniverse server)." >&2
    echo "       Install once on the login node: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Python       : $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown))"
echo "Node.js      : $(node --version)"
echo "Dataset      : $DATASET"
echo "Mode         : $MODE"
echo "Reasoning    : $CLIENT"
echo "Verifier     : charmchat $CHARMCHAT_PROVIDER:$CHARMCHAT_MODEL (servers: $CHARMCHAT_SERVERS, port $CHARMCHAT_PORT)"
echo "Anthropic key: $([ -n "${ANTHROPIC_API_KEY:-}" ] && echo set || echo MISSING)"
echo "OpenAI key   : $([ -n "${OPENAI_API_KEY:-}" ] && echo set || echo MISSING)"
echo "Start time   : $(date)"
echo "──────────────────────────────────────────"

# ── CharmChat: install on first use ────────────────────────────────────────────
if [ ! -d "$CHARMCHAT_DIR" ]; then
    echo "Cloning CharmChat into $CHARMCHAT_DIR ..."
    git clone --depth 1 https://github.com/charm-bdf/charmchat.git "$CHARMCHAT_DIR" || exit 1
fi
if [ ! -d "$CHARMCHAT_DIR/backend/node_modules" ]; then
    echo "Installing CharmChat backend dependencies ..."
    (cd "$CHARMCHAT_DIR/backend" && npm install --no-audit --no-fund) || exit 1
fi

# This job's MCP config: only the servers the agent may use. Prints the Node server dirs to build.
mkdir -p "$CHARMCHAT_DIR/data/config"
NODE_SERVER_DIRS=$("$PYTHON" - "$CHARMCHAT_DIR" "$CHARMCHAT_SERVERS" <<'EOF'
import json, os, sys
root, wanted = sys.argv[1], [s.strip() for s in sys.argv[2].split(",") if s.strip()]
example = json.load(open(os.path.join(root, "mcp-servers.example.json")))["mcpServers"]
missing = [s for s in wanted if s not in example]
if missing:
    sys.exit(f"Unknown MCP servers {missing}; available: {sorted(example)}")
config = {s: example[s] for s in wanted}
json.dump({"mcpServers": config}, open(os.path.join(root, "data/config/mcp-servers.json"), "w"), indent=2)
for spec in config.values():
    if spec["command"] == "node":
        print(os.path.normpath(os.path.join(root, "backend", os.path.dirname(os.path.dirname(spec["args"][0])))))
EOF
) || exit 1
if [ ! -f "$CHARMCHAT_DIR/data/config/settings.json" ]; then
    cp "$CHARMCHAT_DIR/settings.example.json" "$CHARMCHAT_DIR/data/config/settings.json"
fi
for dir in $NODE_SERVER_DIRS; do
    if [ ! -f "$dir/dist/index.js" ]; then
        echo "Building MCP server $dir ..."
        (cd "$dir" && npm install --no-audit --no-fund && npm run build) || exit 1
    fi
done
if [[ ",$CHARMCHAT_SERVERS," == *",tooluniverse,"* ]]; then
    echo "Syncing ToolUniverse MCP server environment ..."
    uv sync --directory "$CHARMCHAT_DIR/mcp-servers/tooluniverse-mcp" || exit 1
fi

# ── CharmChat: start the backend ───────────────────────────────────────────────
CHARMCHAT_LOG="$SLURM_SUBMIT_DIR/slurm_logs/${SLURM_JOB_ID}_charmchat.log"
echo "Starting CharmChat backend (log: $CHARMCHAT_LOG) ..."
(cd "$CHARMCHAT_DIR/backend" && PORT="$CHARMCHAT_PORT" exec ./node_modules/.bin/tsx src/index.ts) \
    > "$CHARMCHAT_LOG" 2>&1 &
CHARMCHAT_PID=$!
trap 'echo "Shutting down CharmChat (PID $CHARMCHAT_PID)..."; pkill -P $CHARMCHAT_PID 2>/dev/null; kill $CHARMCHAT_PID 2>/dev/null; wait $CHARMCHAT_PID 2>/dev/null' EXIT

export CHARMCHAT_API_BASE="http://localhost:${CHARMCHAT_PORT}/api"
echo "Waiting for CharmChat and MCP servers ($CHARMCHAT_SERVERS) ..."
ELAPSED=0
until curl -sf "$CHARMCHAT_API_BASE/mcp" 2>/dev/null | "$PYTHON" -c '
import json, sys
wanted = set(sys.argv[1].split(","))
status = {s["name"]: s["status"] for s in json.load(sys.stdin)["servers"]}
sys.exit(0 if all(status.get(s) == "connected" for s in wanted) else 1)
' "$CHARMCHAT_SERVERS"; do
    if ! kill -0 "$CHARMCHAT_PID" 2>/dev/null; then
        echo "ERROR: CharmChat exited before becoming ready. Last log lines:" >&2
        tail -30 "$CHARMCHAT_LOG" >&2
        exit 1
    fi
    if [ $ELAPSED -ge "$CHARMCHAT_READY_WAIT" ]; then
        echo "ERROR: MCP servers not connected after ${CHARMCHAT_READY_WAIT}s. Status:" >&2
        curl -s "$CHARMCHAT_API_BASE/mcp" | "$PYTHON" -c 'import json,sys; [print("  ", s["name"], s["status"], s.get("error","")) for s in json.load(sys.stdin)["servers"]]' >&2
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "CharmChat ready after ${ELAPSED}s."

# ── Theorem prover ─────────────────────────────────────────────────────────────
export PATH="$SLURM_SUBMIT_DIR/bin:$PATH"

# ── Run the pipeline ───────────────────────────────────────────────────────────
ARGS=(--dataset "$DATASET" --mode "$MODE" --client "$CLIENT" --start "$START"
      --provider "$CHARMCHAT_PROVIDER" --model "$CHARMCHAT_MODEL"
      --servers ${CHARMCHAT_SERVERS//,/ } --output-dir "$OUTPUT_DIR")
[ -n "$NUM_PROBLEMS" ] && ARGS+=(--num-problems "$NUM_PROBLEMS")
[ -n "$QUESTION_TYPES" ] && ARGS+=(--question-types $QUESTION_TYPES)
[ -n "$SKIP_TPTP" ] && ARGS+=(--skip-tptp)
[ "$MODE" = "repair" ] && ARGS+=(--max-retries "$MAX_RETRIES")

export PYTHONPATH="$SLURM_SUBMIT_DIR/..:${PYTHONPATH:-}"
"$PYTHON" "$SLURM_SUBMIT_DIR/../main_pipeline/curebench_benchmark.py" "${ARGS[@]}"
STATUS=$?

# ── Success rates ──────────────────────────────────────────────────────────────
RESULTS_DIR=$(ls -td "$OUTPUT_DIR"/curebench_${MODE}_run_* 2>/dev/null | head -1)
if [ -n "$RESULTS_DIR" ] && [ -f "$RESULTS_DIR/results.json" ]; then
    echo
    "$PYTHON" "$SLURM_SUBMIT_DIR/../main_pipeline/curebench_report.py" "$RESULTS_DIR"
    if [ "$MODE" = "verify" ]; then
        "$PYTHON" "$SLURM_SUBMIT_DIR/../visualize_trees.py" "$RESULTS_DIR" \
            && echo "HTML report: $RESULTS_DIR/report.html"
    fi
else
    echo "WARNING: no finished run found in $OUTPUT_DIR (pipeline exit status $STATUS)." >&2
fi

echo "──────────────────────────────────────────"
echo "End time : $(date)"
exit $STATUS
