"""
Benchmark: compare plain-prompt vs. full pipeline accuracy on a dataset.

Usage:
    python benchmark.py                          # uses dataset.json
    python benchmark.py --dataset my_data.json
    python benchmark.py --max-retries 5
    python benchmark.py --output-dir ./my_bench_run

Output structure:
    benchmark_outputs/<run_id>/
        summary.json              — accuracy stats for both approaches
        results.json              — per-item results
        <item_id>/
            question.txt
            plain_prompt.txt
            plain_response.txt
            plain_answer.txt
            cot.txt
            fol_attempt_1.txt
            verify_attempt_1.txt
            fol_attempt_N.txt     (if retries occurred)
            verify_attempt_N.txt
            fol_final.txt
            tptp_final.tptp
            verify_final.txt
            pipeline_answer.txt
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

from main import run_plain, run_pipeline
from clients import make_client, OpenAILLMClient

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Answer Normalization ──────────────────────────────────────────────────────

def normalize(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip().lower()
    answer = re.sub(r'[$€£,]', '', answer)   # strip currency symbols and commas
    answer = answer.rstrip('.')
    answer = answer.strip()
    try:
        val = float(answer)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return answer


def answers_match(predicted: str, expected: str) -> bool:
    return normalize(predicted) == normalize(expected)


# ─── Benchmark Runner ──────────────────────────────────────────────────────────

def run_benchmark(
    dataset_path: str,
    output_dir: str,
    max_retries: int = 3,
    reasoning_client=None,
    verifier_client=None,
) -> dict:
    with open(dataset_path) as f:
        dataset = json.load(f)

    items = dataset["items"]
    total = len(items)

    plain_correct = 0
    pipeline_correct = 0
    pipeline_passed_verify = 0
    pipeline_passed_tptp = 0
    pipeline_needed_retry = 0   # items where at least one retry was used
    plain_wrong_pipeline_right = 0  # errors caught by pipeline
    plain_right_pipeline_wrong = 0  # regressions introduced by pipeline
    total_attempts = 0
    results = []

    if reasoning_client is None:
        reasoning_client = OpenAILLMClient()
    if verifier_client is None:
        verifier_client = OpenAILLMClient()

    print(f"=== Benchmark: {dataset.get('name', dataset_path)} ===")
    print(f"Items: {total} | Max retries: {max_retries}")
    print(f"Outputs: {output_dir}\n")
    print(f"{'ID':<12} {'Expected':<12} {'Plain':<12} {'Pipeline':<12} {'Plain OK':<10} {'Pipeline OK'}")
    print("-" * 72)

    for item in items:
        item_id = item["id"]
        question = item["question"]
        expected = item["answer"]
        item_dir = os.path.join(output_dir, item_id)
        os.makedirs(item_dir, exist_ok=True)

        # ── Plain prompt ───────────────────────────────────────────────────────
        print(f"\n[{item_id}] Running plain prompt...")
        plain_result = run_plain(question, output_dir=item_dir, reasoning_client=reasoning_client)
        plain_answer = plain_result["answer"]
        plain_ok = answers_match(plain_answer, expected)
        if plain_ok:
            plain_correct += 1

        # ── Full pipeline ──────────────────────────────────────────────────────
        print(f"[{item_id}] Running pipeline...")
        pipeline_result = run_pipeline(
            question, max_retries=max_retries, output_dir=item_dir,
            reasoning_client=reasoning_client, verifier_client=verifier_client,
        )
        pipeline_answer = pipeline_result["answer"]
        pipeline_ok = answers_match(pipeline_answer, expected)
        if pipeline_ok:
            pipeline_correct += 1
        if pipeline_result["passed_verify"]:
            pipeline_passed_verify += 1
        if pipeline_result["passed_tptp"]:
            pipeline_passed_tptp += 1
        if pipeline_result["attempts"] > 1:
            pipeline_needed_retry += 1
        total_attempts += pipeline_result["attempts"]
        if not plain_ok and pipeline_ok:
            plain_wrong_pipeline_right += 1
        if plain_ok and not pipeline_ok:
            plain_right_pipeline_wrong += 1

        # ── Per-item summary ───────────────────────────────────────────────────
        result = {
            "id": item_id,
            "question": question,
            "expected": expected,
            "plain": {
                "answer": plain_answer,
                "correct": plain_ok,
            },
            "pipeline": {
                "answer": pipeline_answer,
                "correct": pipeline_ok,
                "attempts": pipeline_result["attempts"],
                "passed_verify": pipeline_result["passed_verify"],
                "passed_tptp": pipeline_result["passed_tptp"],
            },
        }
        results.append(result)

        plain_marker = "✓" if plain_ok else "✗"
        pipeline_marker = "✓" if pipeline_ok else "✗"
        print(
            f"\n{item_id:<12} {expected:<12} {plain_answer:<12} {pipeline_answer:<12} "
            f"{plain_marker:<10} {pipeline_marker}"
        )

    # ── Aggregate summary ──────────────────────────────────────────────────────
    plain_acc = plain_correct / total if total else 0
    pipeline_acc = pipeline_correct / total if total else 0
    plain_errors = total - plain_correct
    error_catch_rate = plain_wrong_pipeline_right / plain_errors if plain_errors else 0
    regression_rate = plain_right_pipeline_wrong / plain_correct if plain_correct else 0
    verify_rate = pipeline_passed_verify / total if total else 0
    tptp_rate = pipeline_passed_tptp / total if total else 0
    retry_rate = pipeline_needed_retry / total if total else 0
    avg_attempts = total_attempts / total if total else 0

    summary = {
        "dataset": dataset.get("name", dataset_path),
        "run_id": os.path.basename(output_dir),
        "total": total,
        "plain": {
            "correct": plain_correct,
            "accuracy": round(plain_acc, 4),
        },
        "pipeline": {
            "correct": pipeline_correct,
            "accuracy": round(pipeline_acc, 4),
            "passed_verify": pipeline_passed_verify,
            "verify_rate": round(verify_rate, 4),
            "passed_tptp": pipeline_passed_tptp,
            "tptp_rate": round(tptp_rate, 4),
            "needed_retry": pipeline_needed_retry,
            "retry_rate": round(retry_rate, 4),
            "avg_attempts": round(avg_attempts, 2),
        },
        "error_analysis": {
            "plain_errors": plain_errors,
            "errors_caught_by_pipeline": plain_wrong_pipeline_right,
            "error_catch_rate": round(error_catch_rate, 4),
            "regressions": plain_right_pipeline_wrong,
            "regression_rate": round(regression_rate, 4),
        },
    }

    delta = pipeline_acc - plain_acc
    direction = "better" if delta > 0 else "worse" if delta < 0 else "same"

    print("\n" + "=" * 72)
    print(f"{'RESULTS SUMMARY':^72}")
    print("=" * 72)
    print(f"  Total items         : {total}")
    print(f"  Plain prompt        : {plain_correct}/{total} correct  ({plain_acc:.1%})")
    print(f"  Full pipeline       : {pipeline_correct}/{total} correct  ({pipeline_acc:.1%})")
    print(f"  Delta               : {delta:+.1%} ({direction})")
    print()
    print(f"  -- Error Analysis --")
    print(f"  Plain errors        : {plain_errors}/{total} ({1 - plain_acc:.1%} error rate)")
    print(f"  Errors caught       : {plain_wrong_pipeline_right}/{plain_errors}  ({error_catch_rate:.1%} of plain errors fixed by pipeline)")
    print(f"  Regressions         : {plain_right_pipeline_wrong}/{plain_correct}  ({regression_rate:.1%} of plain-correct broken by pipeline)")
    print()
    print(f"  -- Pipeline Internals --")
    print(f"  Passed verify       : {pipeline_passed_verify}/{total} ({verify_rate:.1%})")
    print(f"  Passed TPTP         : {pipeline_passed_tptp}/{total} ({tptp_rate:.1%})")
    print(f"  Needed retry        : {pipeline_needed_retry}/{total} ({retry_rate:.1%})")
    print(f"  Avg attempts/item   : {avg_attempts:.2f}")
    print("=" * 72)

    # ── Write output files ─────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {os.path.join(output_dir, 'summary.json')}")
    print(f"Wrote {os.path.join(output_dir, 'results.json')}")

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark plain prompt vs. full pipeline on a dataset.",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=os.path.join(SCRIPT_DIR, "dataset.json"),
        help="Path to the dataset JSON file (default: dataset.json).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=(
            "Root directory for benchmark outputs. "
            "A timestamped sub-folder is created automatically. "
            "Defaults to benchmark_outputs/ next to this script."
        ),
    )
    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=3,
        help="Max verify/tptp repair attempts per pipeline run (default: 3).",
    )
    parser.add_argument(
        "--client", "-c",
        choices=["openai", "kimi", "deepseek"],
        default="openai",
        help="Reasoning client for CoT/FOL generation (default: openai). "
             "Verification always uses openai.",
    )
    args = parser.parse_args()

    reasoning_client = make_client(args.client)
    verifier_client = OpenAILLMClient()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_dir = args.output_dir or os.path.join(SCRIPT_DIR, "benchmark_outputs")
    output_dir = os.path.join(base_dir, run_id)

    run_benchmark(
        dataset_path=args.dataset,
        output_dir=output_dir,
        max_retries=args.max_retries,
        reasoning_client=reasoning_client,
        verifier_client=verifier_client,
    )
