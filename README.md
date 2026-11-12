# cot-validation

Tools for checking whether an LLM's chain-of-thought (CoT) reasoning actually supports
its answer. A reasoning model writes a CoT, a second model formalizes it into
first-order logic, and the result is checked with an LLM grounding judge plus
automated theorem provers (Vampire and E). Errors found by the checks can be fed back
to the model as repair instructions.

The repo contains three pipelines:

| Directory          | What it does                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| `main_pipeline/`   | CoT, then structured FOL, then verification and TPTP proving, with a repair loop. Math word problems (GSM8K, SVAMP). |
| `rule_pipeline/`   | The same loop for ProofWriter, where the CoT must reason only from the given facts and rules.  |
| `graph_pipeline/`  | Experimental. Turns a CoT into an editable inference graph and checks each step individually.  |

## Setup

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai anthropic boto3 python-dotenv datasets requests graphviz
```

`visualize_trees.py` also needs the Graphviz system package (`brew install graphviz`
on macOS).

### Theorem provers

- **Vampire** is used for arithmetic checks and TPTP proving. It is looked up in this
  order: `$VAMPIRE_PATH`, `vampire` on `PATH`, then `tools/vampire`. `tools/` is
  gitignored, so download a binary for your platform from the
  [Vampire releases](https://github.com/vprover/vampire/releases) and put it there.
  If Vampire is missing, arithmetic is checked with exact Python fractions instead.
- **E** (`eprover`) is used for TPTP syntax checking and proving. The binary in `bin/`
  is a Linux x86-64 build for the cluster; on macOS install it with `brew install eprover`.

### API keys

Put keys in a `.env` file at the repo root (it is gitignored). Only the clients you use
need keys.

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
KIMI_API_KEY=...
KIMI_BASE_URL=...
BEDROCK_API_KEY=...          # or AWS_BEARER_TOKEN_BEDROCK / standard AWS credentials
```

## LLM clients

All pipelines take a client name, defined in `main_pipeline/clients.py`:

| Name                  | Backend                    | Default model (env var to override)              |
| --------------------- | -------------------------- | ------------------------------------------------ |
| `openai`              | OpenAI Responses API       | `gpt-5-nano`                                     |
| `claude`, `anthropic` | Anthropic Messages API     | `claude-opus-5-5` (`ANTHROPIC_MODEL`)            |
| `bedrock`             | Claude on Amazon Bedrock   | `anthropic.claude-sonnet-5-5` (`BEDROCK_MODEL`)  |
| `deepseek`            | DeepSeek                   | `deepseek-chat` (`DEEPSEEK_MODEL`)               |
| `kimi`                | Moonshot / Kimi            | `moonshot-v1-8k` (`KIMI_MODEL`)                  |
| `vllm`                | Local vLLM server          | `Qwen/Qwen2.5-1.5B-Instruct` (`VLLM_MODEL`, `VLLM_BASE_URL`) |

Every call records token usage and estimated USD cost; benchmark runs print a cost
summary at the end.

## Data

Datasets live in `data/` as `{"name": ..., "items": [{"id", "question", "answer"}, ...]}`.

- `data/dataset.json`: GSM8K test split (1,319 items)
- `data/math_word_problems.json`: SVAMP
- `data/proofwriter.json`: ProofWriter (items have `theory`, rules and a statement to label)

To regenerate them:

```bash
cd data
python load_data.py --gsm8k --split test -o dataset.json
python load_data.py --proofwriter --split test --limit 100 -o proofwriter.json
```

## Running

Run everything from the repo root with `PYTHONPATH=.` so the `main_pipeline` package
resolves.

### Main pipeline (single question)

```bash
PYTHONPATH=. python main_pipeline/main.py -q "Sam has 3 boxes of 12 pencils..." -o out/sam
PYTHONPATH=. python main_pipeline/main.py --plain -q "..."     # direct answer, no CoT/FOL
```

The steps are:

1. **CoT.** The reasoning client answers with a syllogism-style argument ending in
   `The answer is: X`.
2. **Structured FOL.** The verifier client converts the CoT into a fixed vocabulary
   (at most 5 predicates) and four labelled sections: `FACT-n`, `OP-n` (arithmetic using
   the built-in `Sum`/`Difference`/`Product`/`Quotient`), `RULE-n` and `INF-n`.
3. **Verification** (`verify.py`). An LLM judge checks that each fact is grounded in the
   question and each rule is plausible.
4. **Arithmetic** (`arithmetic.py`). Each `OP` line is compiled to TPTP-TFF over `$rat`
   and Vampire proves either the claim or its negation.
5. **TPTP** (`tptp.py`). The FOL is converted to TPTP and each inference is checked with
   E or Vampire.

If verification or arithmetic fails, the CoT is regenerated using the error report. If
the prover fails, only the FOL is repaired. This repeats up to `--max-retries` times
(default 5); if it never passes, the answer is reported as unverified.

### Benchmarks

```bash
# Accuracy of a plain prompt vs. the full repair pipeline
PYTHONPATH=. python main_pipeline/benchmark.py --dataset data/dataset.json \
    -n 20 --client openai --verifier-client openai

# Verify-only: no repair; measures how well verification flags wrong CoTs
PYTHONPATH=. python main_pipeline/verify_benchmark.py --dataset data/dataset.json \
    -n 20 --client openai --verifier claude [--skip-tptp]

# HTML report with derivation trees for a verify-only run
python visualize_trees.py benchmark_outputs/verify_run_<timestamp>
```

`benchmark.py` writes to `main_pipeline/benchmark_outputs/run_<timestamp>/` and
`verify_benchmark.py` to `benchmark_outputs/verify_run_<timestamp>/`. Each run folder
has `summary.json`, `results.json` and one subfolder per item with the question, CoT,
FOL, verification and prover reports. The verify-only summary reports a confusion
matrix: correct/incorrect answers against verified/flagged.

### Optimizing the formal-verification prompts (DSPy)

`formal_dspy/` uses [DSPy](https://dspy.ai) GEPA to rewrite the two prompts that feed the
provers: FOL extraction (`STRUCTURED_FOL_TEMPLATE`) and FOL-to-TPTP conversion
(`TPTP_TEMPLATE`). The fact/rule judge and the graph pipeline are not touched. Training data
is every saved CoT in `benchmark_outputs/*_run_*/` with its correct/incorrect label, split by
question into train, dev and test. Requires `pip install dspy`.

```bash
python -m formal_dspy.optimize --verifier bedrock --reflection bedrock \
    --reflection-model anthropic.claude-opus-5-5 --auto light
python -m formal_dspy.optimize --eval-only                    # score the current prompts
PYTHONPATH=. python main_pipeline/verify_benchmark.py ... \
    --formal-program formal_dspy/runs/<run>/program.json      # use the optimized prompts
```

Each formalization is checked by the existing provers and scored:

| CoT answer | proved | refuted (an OP is wrong or an inference is not proved) | broken (syntax error, timeout, no inferences) |
| ---------- | ------ | ------------------------------------------------------ | --------------------------------------------- |
| correct    | 1      | 0                                                      | 0                                             |
| incorrect  | 0.5    | 1                                                      | 0                                             |

A number in a FACT or OP formula that is not in the question or the CoT scores 0, so the
optimizer cannot learn to correct or invent arithmetic. GEPA also receives the prover
reports as written feedback. Runs go to `formal_dspy/runs/formal_<timestamp>/`: `report.json`
(baseline and optimized test scores, splits, cost), `program.json`, and the optimized
instructions as text. On the cluster use `slurm_scripts/run_formal_optimize.slurm` (CPU only).

### ProofWriter pipeline

```bash
PYTHONPATH=. python rule_pipeline/rule_pipeline.py -f data/proofwriter.json -n 20 --client openai
```

Answers are True, False or Unknown. Facts must be grounded in the problem and rules in
the given ruleset, rather than in commonsense. Output goes to
`rule_pipeline/proofwriter_outputs/run_<timestamp>/`.

### Graph pipeline (experimental)

```bash
# Build a graph from an existing benchmark item (uses question.txt and cot.txt)
python -m graph_pipeline.run --item benchmark_outputs/verify_run_20261001_150731/item_0023

# From your own files
python -m graph_pipeline.run --question-file q.txt --cot-file cot.txt -o graph_outputs/mine

# Re-check and re-render a saved graph without any LLM calls
python -m graph_pipeline.run --graph graph_pipeline/examples/item_0023.json
```

The default client is `bedrock` (Claude Opus); change it with `--client`. LLM agents
edit a single graph in order:

1. **Structurer** turns the CoT into claims (facts, rules, closures) and inference steps.
2. **Classifier** records where each non-derived claim comes from (the problem, math,
   definitions, commonsense, a stipulation, and so on).
3. **Implicit** adds unstated assumptions the reasoning relies on.
4. An **encoder** fills in structured content so the graph can be compiled.

Every edit goes through `ops.apply_ops` and is logged. Each step is then checked
deterministically (`match.py`): its rule is unified with its premises, arithmetic is
evaluated with exact fractions, and a truth table decides entailment. Step statuses are
`matched`, `gap`, `arithmetic_mismatch` (the CoT's own arithmetic is wrong) or `unchecked`.

Finally (`prove.py`), each step is compiled to its own TFF problem (premises and warrant
as axioms, conclusion as the conjecture) and given to Vampire in portfolio mode. Each step
gets a prover status: `proved`, `not_proved`, `contradictory_premises`, `timeout`,
`unknown`, `error` or `skipped` (Vampire not found). The graph is `VERIFIED` only if every
step is proved. E is not used here because it does not support `$rat` arithmetic. Use
`--no-prover` to skip this check, or `--prover-timeout N` to change the per-step limit
(default 20 s). With arithmetic, Vampire rarely finds countermodels, so an invalid step
usually shows up as `timeout` rather than `not_proved`.

Outputs, written to `graph_outputs/<item>/` by default:

- `graph.json`: the graph
- `oplog.jsonl`: every edit
- `graph.tptp`: a TFF translation
- `prover_report.txt`: the verdict and each step's prover status
- `argument.txt`: the argument rendered back into English
- `graph.html`: an interactive view (loads Cytoscape from a CDN)

#### Graph benchmark

```bash
python -m graph_pipeline.benchmark --dataset data/dataset.json -n 20 \
    --client vllm --verifier bedrock [--start 0] [--prover-timeout 20]
```

For each item the reasoning client (`--client`) writes a CoT, the graph agents
(`--verifier`) build the graph, and Vampire proves each step. An answer is **verified**
only if every step is proved, no assumption is `unjustified` or unclassified, and a final
conclusion of the graph states the CoT's answer. Verified is compared with correct, as in
the verify-only benchmark. Output goes to `benchmark_outputs/graph_run_<timestamp>/`:
`summary.json`, `results.json` (rewritten after every item), `index.html` (one row per
item linking to its `graph.html`), and one folder per item with the usual graph outputs.

The longer-term design (assumption tiers and costs, an agent
orchestrator) is described in `inference_graph_plan.txt`.

## Running on a Slurm cluster

`slurm_scripts/` has job scripts for the FASRC cluster. Submit them from inside
`slurm_scripts/` and override settings with environment variables:

```bash
cd slurm_scripts
sbatch --export=ALL,CLIENT=bedrock,VERIFIER=bedrock,NUM_PROBLEMS=50 run_verify_benchmark.slurm
sbatch --export=ALL,VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct run_verify_benchmark_vllm.slurm

# Graph benchmark: Qwen on vLLM writes the CoTs, Claude Opus on Bedrock builds the graphs
sbatch --export=ALL,NUM_PROBLEMS=50,VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct run_graph_benchmark_vllm.slurm
```

`run_graph_benchmark_vllm.slurm` also takes `DATASET`, `START`, `OUTPUT_DIR`,
`VERIFIER_CLIENT` (default `bedrock`), `BEDROCK_MODEL`, `ANTHROPIC_MODEL` and
`PROVER_TIMEOUT`. On first run it downloads the Linux Vampire build into `tools/linux/`
(set `VAMPIRE_PATH` to use your own); if compute nodes have no internet, unzip
`vampire-Linux-X64.zip` from the Vampire releases there yourself.

The `*_vllm.slurm` scripts start a vLLM server on the allocated GPU, wait for it to
come up, use it as the reasoning client, and shut it down when the job exits. They
expect a conda environment named `vllm-inference`. The partition and account names
are cluster-specific, so edit them for your site. Logs go to `slurm_logs/`.

## Repository layout

```
main_pipeline/      CoT -> FOL -> verify -> TPTP pipeline, benchmarks, LLM clients
rule_pipeline/      ProofWriter variant of the main pipeline
graph_pipeline/     inference-graph builder, step matcher, TFF/English/HTML renderers
data/               datasets and the loader script
slurm_scripts/      cluster job scripts
visualize_trees.py  HTML derivation-tree report for verify-only runs
bin/                E prover binary (Linux)
benchmark_outputs/, graph_outputs/, proofwriter_outputs/   run outputs
```
