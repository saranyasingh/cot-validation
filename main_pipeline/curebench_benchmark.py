"""
Run the main pipeline on CureBench, with a CharmChat agent as the verifier client.

The CharmChat backend (https://github.com/charm-bdf/charmchat) must be running, by default on
localhost:3001, with the MCP servers you want the agent to use (ToolUniverse has the FDA label
tools). The verifier handles FOL extraction, the tool-backed fact/rule check
(main_pipeline/curebench.py), and TPTP conversion. The reasoning client writes the CoT.

Modes:
  verify  (default) — verify_benchmark.run_verify_benchmark: one CoT per question, verified once,
                      reported as a confusion matrix of correct/incorrect vs. verified/flagged.
  repair            — main.run_pipeline: the full verify → repair loop, where the error report
                      sent back to the reasoning model includes the tool evidence.

Usage:
    python main_pipeline/curebench_benchmark.py --num-problems 10
    python main_pipeline/curebench_benchmark.py --client vllm --servers tooluniverse --skip-tptp
    python main_pipeline/curebench_benchmark.py --mode repair --max-retries 3 -n 20
"""

import argparse
import json
import os
from datetime import datetime

from main_pipeline.clients import (
    CharmChatClient, make_client,
    start_cost_log, set_cost_context, call_log_mark, summarize_costs, print_cost_summary,
)
from main_pipeline.curebench import CUREBENCH_TASK, format_question
from main_pipeline.main import run_pipeline
from main_pipeline.verify_benchmark import run_verify_benchmark

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "..", "data", "curebench_valset_phase1.jsonl")


def load_curebench(path: str, start: int = 0, num_problems: int = None, question_types=None) -> list[dict]:
    """Read the CureBench JSONL into verify_benchmark's item format (id, question, answer)."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if question_types:
        rows = [r for r in rows if r["question_type"] in question_types]
    rows = rows[start:]
    if num_problems is not None:
        rows = rows[:num_problems]
    missing = [r["id"] for r in rows if not r.get("correct_answer")]
    if missing:
        raise ValueError(f"{len(missing)} items have no correct_answer (is this the test set?): {missing[:5]}")
    return [
        {
            "id": r["id"],
            "question": format_question(r),
            "answer": r["correct_answer"],
            "question_type": r["question_type"],
        }
        for r in rows
    ]


def run_repair_benchmark(items, output_dir, reasoning_client, verifier_client, max_retries):
    """Full repair loop per item. Reports accuracy of the first CoT vs. the pipeline's final answer."""
    results = []
    run_mark = call_log_mark()
    start_cost_log(os.path.join(output_dir, "api_calls.jsonl"))

    for item in items:
        item_dir = os.path.join(output_dir, item["id"])
        print(f"\n[{item['id']}] Running pipeline...")
        set_cost_context(item_id=item["id"])
        item_mark = call_log_mark()
        result = run_pipeline(
            item["question"], max_retries=max_retries, output_dir=item_dir,
            reasoning_client=reasoning_client, verifier_client=verifier_client, task=CUREBENCH_TASK,
        )
        with open(os.path.join(item_dir, "cot.txt")) as f:
            first_answer = CUREBENCH_TASK.extract_answer(f.read())
        results.append({
            "id": item["id"],
            "question_type": item["question_type"],
            "expected": item["answer"],
            "first_cot_answer": first_answer,
            "first_cot_correct": first_answer == item["answer"],
            "final_cot_answer": CUREBENCH_TASK.extract_answer(result["cot_text"]),
            "pipeline_answer": result["answer"],
            "pipeline_correct": result["answer"] == item["answer"],
            "attempts": result["attempts"],
            "passed_verify": result["passed_verify"],
            "passed_tptp": result["passed_tptp"],
            "cost": summarize_costs(item_mark),
        })
        r = results[-1]
        print(f"\n{r['id']:<14} expected {r['expected']}  first CoT {r['first_cot_answer'] or '-'}  "
              f"pipeline {r['pipeline_answer']!r}  attempts {r['attempts']}")

    set_cost_context()
    total = len(results)
    first_correct = sum(r["first_cot_correct"] for r in results)
    final_correct = sum(r["pipeline_correct"] for r in results)
    verified = sum(r["passed_verify"] and r["passed_tptp"] for r in results)
    verified_correct = sum(r["passed_verify"] and r["passed_tptp"] and r["pipeline_correct"] for r in results)
    summary = {
        "dataset": "curebench",
        "run_id": os.path.basename(output_dir),
        "mode": "repair",
        "total": total,
        "max_retries": max_retries,
        "reasoning_model": reasoning_client.model,
        "verifier_model": f"charmchat:{verifier_client.provider}:{verifier_client.model}",
        "verifier_blocked_servers": verifier_client.blocked_servers,
        "first_cot": {"correct": first_correct, "accuracy": round(first_correct / total, 4) if total else 0},
        "pipeline": {
            "correct": final_correct,
            "accuracy": round(final_correct / total, 4) if total else 0,
            "verified": verified,
            "verified_and_correct": verified_correct,
            "verified_precision": round(verified_correct / verified, 4) if verified else 0,
        },
        "cost": summarize_costs(run_mark),
    }

    print("\n" + "=" * 74)
    print(f"  Items                      : {total}")
    print(f"  First CoT correct          : {first_correct}/{total}")
    print(f"  Pipeline answer correct    : {final_correct}/{total}")
    print(f"  Verified (passed all)      : {verified}/{total}  ({verified_correct} of them correct)")
    print_cost_summary(summary["cost"])
    print("=" * 74)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {os.path.join(output_dir, 'summary.json')}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CoT → FOL → verify pipeline on CureBench via CharmChat.")
    parser.add_argument("--dataset", "-d", default=DEFAULT_DATASET,
                        help="CureBench JSONL with correct_answer (default: data/curebench_valset_phase1.jsonl).")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Root output directory. A timestamped sub-folder is created automatically.")
    parser.add_argument("--start", type=int, default=0, help="Index of the first question to run.")
    parser.add_argument("--num-problems", "-n", type=int, default=None, help="Number of questions (default: all).")
    parser.add_argument("--question-types", nargs="+", default=None,
                        choices=["multi_choice", "open_ended_multi_choice", "open_ended"],
                        help="Only run these question types.")
    parser.add_argument("--mode", choices=["verify", "repair"], default="verify")
    parser.add_argument("--client", "-c",
                        choices=["openai", "claude", "kimi", "deepseek", "vllm", "bedrock", "charmchat"],
                        default="openai", help="Reasoning client for CoT generation (default: openai).")
    parser.add_argument("--provider", default=None,
                        help="CharmChat provider for the verifier agent (default: CHARMCHAT_PROVIDER or backend settings).")
    parser.add_argument("--model", default=None,
                        help="CharmChat model for the verifier agent (default: CHARMCHAT_MODEL or backend settings).")
    parser.add_argument("--servers", nargs="+", default=None,
                        help="MCP servers the verifier agent may use; all others are blocked "
                             "(default: CHARMCHAT_SERVERS, else the backend's blockedServers setting).")
    parser.add_argument("--skip-tptp", action="store_true", help="Verify mode: skip the TPTP/prover step.")
    parser.add_argument("--max-retries", "-r", type=int, default=5, help="Repair mode: max attempts (default: 5).")
    args = parser.parse_args()

    run_id = datetime.now().strftime(f"curebench_{args.mode}_run_%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir or os.path.join(SCRIPT_DIR, "..", "benchmark_outputs"), run_id)
    os.makedirs(output_dir, exist_ok=True)

    items = load_curebench(args.dataset, args.start, args.num_problems, args.question_types)
    dataset_path = os.path.join(output_dir, "dataset.json")
    with open(dataset_path, "w") as f:
        json.dump({"name": f"curebench ({os.path.basename(args.dataset)})", "items": items}, f, indent=2)

    reasoning_client = make_client(args.client)
    verifier_client = CharmChatClient(provider=args.provider, model=args.model, servers=args.servers,
                                      transcript_dir=output_dir)
    print(f"CharmChat verifier: {verifier_client.provider}:{verifier_client.model} "
          f"(blocked servers: {verifier_client.blocked_servers or 'none'})")

    if args.mode == "verify":
        run_verify_benchmark(
            dataset_path=dataset_path,
            output_dir=output_dir,
            reasoning_client=reasoning_client,
            verifier_client=verifier_client,
            skip_tptp=args.skip_tptp,
            reasoning_client_name=args.client,
            verifier_client_name=f"charmchat:{verifier_client.provider}:{verifier_client.model}",
            task=CUREBENCH_TASK,
        )
    else:
        run_repair_benchmark(items, output_dir, reasoning_client, verifier_client, args.max_retries)
