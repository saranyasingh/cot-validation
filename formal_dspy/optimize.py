"""
Optimize the formal-verification prompts (FOL extraction and TPTP conversion) with GEPA,
using saved benchmark runs as labelled data: each item's question, CoT and whether its
answer was correct. Items are split by question into train / dev / test, so the test
score is on problems GEPA never saw.

Usage:
  python -m formal_dspy.optimize --verifier bedrock --auto light
  python -m formal_dspy.optimize --verifier bedrock --max-metric-calls 300 --num-threads 8
  python -m formal_dspy.optimize --eval-only                                   # baseline only
  python -m formal_dspy.optimize --eval-only --program formal_dspy/runs/<run>/program.json

Use the optimized prompts in the verify-only benchmark with
  main_pipeline/verify_benchmark.py --formal-program formal_dspy/runs/<run>/program.json
"""

import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime

import dspy

from formal_dspy.lm import CLIENTS, REPO_ROOT, make_lm
from formal_dspy.program import FormalVerifier, feedback_metric, metric, score_prediction
from main_pipeline.clients import call_log_mark, print_cost_summary, start_cost_log, summarize_costs
from main_pipeline.tptp import find_prover

DEFAULT_RUNS = [os.path.join(REPO_ROOT, "benchmark_outputs", "verify_run_*"),
                os.path.join(REPO_ROOT, "benchmark_outputs", "graph_run_*")]


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_examples(patterns: list[str]) -> list[dspy.Example]:
    """One example per saved CoT that has a results.json entry and a cot.txt."""
    examples = []
    run_dirs = sorted({d for p in patterns for d in glob.glob(p) if os.path.isdir(d)})
    for run_dir in run_dirs:
        results_path = os.path.join(run_dir, "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            results = json.load(f)
        for r in results:
            cot_path = os.path.join(run_dir, r["id"], "cot.txt")
            if not os.path.exists(cot_path):
                continue
            with open(cot_path) as f:
                cot_text = f.read()
            examples.append(dspy.Example(
                question=r["question"], cot_text=cot_text, cot_answer=r.get("cot_answer", ""),
                expected=str(r["expected"]), cot_correct=bool(r["cot_correct"]),
                item_id=r["id"], run=os.path.basename(run_dir),
            ).with_inputs("cot_text"))
    return examples


def split_by_question(examples, seed: int, fractions=(0.5, 0.25)):
    """Train / dev / test with every CoT for the same question in the same split."""
    questions = sorted({" ".join(ex.question.split()) for ex in examples})
    random.Random(seed).shuffle(questions)
    n_train = int(len(questions) * fractions[0])
    n_dev = int(len(questions) * fractions[1])
    which = {q: "train" if i < n_train else "dev" if i < n_train + n_dev else "test"
             for i, q in enumerate(questions)}
    splits = {"train": [], "dev": [], "test": []}
    for ex in examples:
        splits[which[" ".join(ex.question.split())]].append(ex)
    return splits["train"], splits["dev"], splits["test"]


def describe(name: str, examples) -> str:
    correct = sum(ex.cot_correct for ex in examples)
    return f"{name}: {len(examples)} CoTs ({correct} correct, {len(examples) - correct} incorrect)"


# ─── Evaluation ───────────────────────────────────────────────────────────────

def summarize_eval(result) -> dict:
    """Verdict counts for correct and incorrect CoTs, plus the mean metric score."""
    counts = {"correct": {}, "incorrect": {}}
    unfaithful = 0
    rows = []
    for ex, pred, score in result.results:
        verdict = getattr(pred, "verdict", None) or "failed"
        if score_prediction(ex, pred)[1].startswith("unfaithful"):
            unfaithful += 1
        side = counts["correct" if ex.cot_correct else "incorrect"]
        side[verdict] = side.get(verdict, 0) + 1
        rows.append({"run": ex.run, "id": ex.item_id, "cot_correct": ex.cot_correct,
                     "verdict": verdict, "score": score})
    return {"mean_score": round(result.score / 100, 4), "verdicts": counts,
            "unfaithful": unfaithful, "items": rows}


def print_eval(name: str, summary: dict):
    c, i = summary["verdicts"]["correct"], summary["verdicts"]["incorrect"]
    print(f"\n  {name}: mean score {summary['mean_score']:.3f}")
    print(f"    correct CoTs   : proved {c.get('proved', 0)}, refuted {c.get('refuted', 0)}, "
          f"broken {c.get('broken', 0)}, failed {c.get('failed', 0)}")
    print(f"    incorrect CoTs : refuted {i.get('refuted', 0)}, proved {i.get('proved', 0)}, "
          f"broken {i.get('broken', 0)}, failed {i.get('failed', 0)}")
    print(f"    unfaithful numbers in {summary['unfaithful']} formalizations")


def evaluate(program, examples, num_threads: int) -> dict:
    evaluator = dspy.Evaluate(devset=examples, metric=metric, num_threads=num_threads,
                              display_progress=True, provide_traceback=True)
    return summarize_eval(evaluator(program))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEPA-optimize the formal-verification prompts.")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS,
                        help="benchmark run directories or globs (default: all verify_run_* and graph_run_*)")
    parser.add_argument("--verifier", choices=CLIENTS, default="bedrock",
                        help="client that runs the formalization prompts (default: bedrock)")
    parser.add_argument("--verifier-model", default=None, help="override the verifier client's model")
    parser.add_argument("--reflection", choices=CLIENTS, default="bedrock",
                        help="client GEPA uses to rewrite the prompts (default: bedrock)")
    parser.add_argument("--reflection-model", default=None, help="override the reflection client's model")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
    budget.add_argument("--max-metric-calls", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true", help="evaluate on the test split and stop")
    parser.add_argument("--program", default=None, help="start from a saved program.json")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "formal_dspy", "runs"))
    args = parser.parse_args()
    if not args.eval_only and not (args.auto or args.max_metric_calls):
        args.auto = "light"

    if find_prover()[0] is None:
        sys.exit("No TPTP prover found (eprover or vampire on PATH); the inference check cannot run.")

    examples = load_examples(args.runs)
    if not examples:
        sys.exit(f"No labelled CoTs found in {args.runs}")
    train, dev, test = split_by_question(examples, args.seed)

    run_dir = os.path.join(args.output_dir, datetime.now().strftime("formal_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    start_cost_log(os.path.join(run_dir, "api_calls.jsonl"))
    mark = call_log_mark()

    task_lm = make_lm(args.verifier, args.verifier_model)
    dspy.configure(lm=task_lm)
    print(f"=== Formal-verification prompt optimization ===")
    print(f"Verifier: {args.verifier} ({task_lm.model})")
    for line in (describe("train", train), describe("dev", dev), describe("test", test)):
        print("  " + line)
    print(f"Outputs: {run_dir}")

    program = FormalVerifier()
    if args.program:
        program.load(args.program)

    report = {"args": vars(args), "verifier_model": task_lm.model,
              "splits": {name: [f"{ex.run}/{ex.item_id}" for ex in split]
                         for name, split in (("train", train), ("dev", dev), ("test", test))}}

    print("\nEvaluating the starting prompts on the test split...")
    report["baseline_test"] = evaluate(program, test, args.num_threads)
    print_eval("starting prompts (test)", report["baseline_test"])

    if not args.eval_only:
        reflection_lm = make_lm(args.reflection, args.reflection_model)
        report["reflection_model"] = reflection_lm.model
        print(f"\nRunning GEPA (reflection: {args.reflection} ({reflection_lm.model}), "
              f"budget: {args.auto or f'{args.max_metric_calls} metric calls'})...")
        gepa = dspy.GEPA(metric=feedback_metric, auto=args.auto, max_metric_calls=args.max_metric_calls,
                         reflection_lm=reflection_lm, num_threads=args.num_threads, seed=args.seed,
                         track_stats=True, log_dir=os.path.join(run_dir, "gepa"))
        optimized = gepa.compile(program, trainset=train, valset=dev)
        optimized.save(os.path.join(run_dir, "program.json"))
        for name, predictor in optimized.named_predictors():
            with open(os.path.join(run_dir, f"{name}_instructions.txt"), "w") as f:
                f.write(predictor.signature.instructions + "\n")

        print("\nEvaluating the optimized prompts on the test split...")
        report["optimized_test"] = evaluate(optimized, test, args.num_threads)
        print_eval("starting prompts (test)", report["baseline_test"])
        print_eval("optimized prompts (test)", report["optimized_test"])

    report["cost"] = summarize_costs(mark)
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print()
    print_cost_summary(report["cost"])
    print(f"\nWrote {run_dir}/report.json" + ("" if args.eval_only else
          f", program.json and the optimized instructions"))


if __name__ == "__main__":
    main()
