"""
Verify-only benchmark: generate CoT, verify it (fact/rule + TPTP), flag as incorrect if
verification fails. No repair loop. Measures how well verification catches wrong answers
vs. incorrectly rejecting right ones.

Stats reported:
  - True positives  (errors correctly identified): CoT was wrong AND verification flagged it
  - False positives (correct but flagged wrong):   CoT was right AND verification flagged it
  - False negatives (errors missed):               CoT was wrong AND verification passed

Usage:
    python verify_benchmark.py
    python verify_benchmark.py --dataset data/dataset.json
    python verify_benchmark.py --num-problems 20 --client kimi
    python verify_benchmark.py --output-dir ./my_run --skip-tptp
"""

import argparse
import json
import os
from datetime import datetime

from main_pipeline.main import COT_PROMPT, STRUCTURED_FOL_TEMPLATE, extract_answer
from main_pipeline.verify import verify_fol
from main_pipeline.tptp import convert_and_check
from main_pipeline.clients import make_client, OpenAILLMClient
from main_pipeline.benchmark import answers_match

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ─── Single-item runner ────────────────────────────────────────────────────────

def run_verify_only(
    question: str,
    output_dir: str = None,
    reasoning_client=None,
    verifier_client=None,
    skip_tptp: bool = False,
) -> dict:
    """
    Generate CoT, convert to FOL, run verification once (no repair).

    Returns:
        cot_text        — raw CoT output
        cot_answer      — answer extracted from CoT
        structured_fol  — FOL extracted from CoT
        verify_flagged  — True if verify.py OR tptp found errors
        verify_report   — text report from verify.py
        tptp_report     — text report from tptp (empty string if skipped or not reached)
        passed_verify   — whether fact/rule check passed
        passed_tptp     — whether TPTP check passed (None if skipped)
    """
    if reasoning_client is None:
        reasoning_client = OpenAILLMClient()
    if verifier_client is None:
        verifier_client = OpenAILLMClient()

    # Stage 1: CoT
    cot_text = reasoning_client.complete(COT_PROMPT.format(question=question))
    cot_answer = extract_answer(cot_text)

    if output_dir:
        _write(os.path.join(output_dir, "question.txt"), question)
        _write(os.path.join(output_dir, "cot.txt"), cot_text)

    # Stage 2: FOL extraction
    structured_fol = verifier_client.complete(STRUCTURED_FOL_TEMPLATE.format(cot_text=cot_text))

    if output_dir:
        _write(os.path.join(output_dir, "fol.txt"), structured_fol)

    # Stage 3: fact/rule verification
    verify_errors, verify_report = verify_fol(question, structured_fol, client=verifier_client)

    if output_dir:
        _write(os.path.join(output_dir, "verify_report.txt"), verify_report)

    if verify_errors:
        return {
            "cot_text": cot_text,
            "cot_answer": cot_answer,
            "structured_fol": structured_fol,
            "verify_flagged": True,
            "verify_report": verify_report,
            "tptp_report": "",
            "passed_verify": False,
            "passed_tptp": None if skip_tptp else False,
        }

    # Stage 4: TPTP (optional)
    if skip_tptp:
        return {
            "cot_text": cot_text,
            "cot_answer": cot_answer,
            "structured_fol": structured_fol,
            "verify_flagged": False,
            "verify_report": verify_report,
            "tptp_report": "",
            "passed_verify": True,
            "passed_tptp": None,
        }

    tptp_text, tptp_errors, tptp_report = convert_and_check(structured_fol, client=verifier_client)

    if output_dir:
        _write(os.path.join(output_dir, "tptp.tptp"), tptp_text)
        _write(os.path.join(output_dir, "tptp_report.txt"), tptp_report)

    return {
        "cot_text": cot_text,
        "cot_answer": cot_answer,
        "structured_fol": structured_fol,
        "verify_flagged": tptp_errors,
        "verify_report": verify_report,
        "tptp_report": tptp_report,
        "passed_verify": True,
        "passed_tptp": not tptp_errors,
    }


# ─── Benchmark runner ──────────────────────────────────────────────────────────

def run_verify_benchmark(
    dataset_path: str,
    output_dir: str,
    num_problems: int = None,
    reasoning_client=None,
    verifier_client=None,
    skip_tptp: bool = False,
) -> dict:
    with open(dataset_path) as f:
        dataset = json.load(f)

    items = dataset["items"]
    if num_problems is not None:
        items = items[:num_problems]
    total = len(items)

    if reasoning_client is None:
        reasoning_client = OpenAILLMClient()
    if verifier_client is None:
        verifier_client = OpenAILLMClient()

    # Counters
    cot_correct_count = 0
    flagged_count = 0
    true_positives = 0   # CoT wrong + flagged
    false_positives = 0  # CoT right + flagged
    false_negatives = 0  # CoT wrong + not flagged
    true_negatives = 0   # CoT right + not flagged

    results = []

    print(f"=== Verify-Only Benchmark: {dataset.get('name', dataset_path)} ===")
    print(f"Items: {total} | Skip TPTP: {skip_tptp}")
    print(f"Outputs: {output_dir}\n")
    print(f"{'ID':<12} {'Expected':<12} {'CoT Ans':<12} {'CoT OK':<10} {'Flagged':<10} {'Outcome'}")
    print("-" * 74)

    for item in items:
        item_id = item["id"]
        question = item["question"]
        expected = item["answer"]
        item_dir = os.path.join(output_dir, item_id)

        print(f"\n[{item_id}] Running verify-only pipeline...")
        result = run_verify_only(
            question,
            output_dir=item_dir,
            reasoning_client=reasoning_client,
            verifier_client=verifier_client,
            skip_tptp=skip_tptp,
        )

        cot_answer = result["cot_answer"]
        flagged = result["verify_flagged"]
        cot_ok = answers_match(cot_answer, expected)

        if cot_ok:
            cot_correct_count += 1
        if flagged:
            flagged_count += 1

        if not cot_ok and flagged:
            true_positives += 1
            outcome = "TP"
        elif cot_ok and flagged:
            false_positives += 1
            outcome = "FP"
        elif not cot_ok and not flagged:
            false_negatives += 1
            outcome = "FN"
        else:
            true_negatives += 1
            outcome = "TN"

        entry = {
            "id": item_id,
            "question": question,
            "expected": expected,
            "cot_answer": cot_answer,
            "cot_correct": cot_ok,
            "verify_flagged": flagged,
            "passed_verify": result["passed_verify"],
            "passed_tptp": result["passed_tptp"],
            "outcome": outcome,
        }
        results.append(entry)

        cot_marker = "✓" if cot_ok else "✗"
        flag_marker = "flagged" if flagged else "passed"
        print(
            f"\n{item_id:<12} {expected:<12} {cot_answer:<12} {cot_marker:<10} {flag_marker:<10} {outcome}"
        )

    # ── Aggregate stats ────────────────────────────────────────────────────────
    cot_errors = total - cot_correct_count
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    summary = {
        "dataset": dataset.get("name", dataset_path),
        "run_id": os.path.basename(output_dir),
        "total": total,
        "skip_tptp": skip_tptp,
        "cot": {
            "correct": cot_correct_count,
            "incorrect": cot_errors,
            "accuracy": round(cot_correct_count / total, 4) if total else 0,
        },
        "verification": {
            "flagged": flagged_count,
            "passed": total - flagged_count,
        },
        "error_detection": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
    }

    print("\n" + "=" * 74)
    print(f"{'RESULTS SUMMARY':^74}")
    print("=" * 74)
    print(f"  Total items              : {total}")
    print(f"  CoT correct              : {cot_correct_count}/{total} ({cot_correct_count/total:.1%})")
    print(f"  CoT incorrect            : {cot_errors}/{total}")
    print()
    print(f"  -- Verification Error Detection --")
    print(f"  Errors correctly flagged : {true_positives}/{cot_errors}  (true positives)")
    print(f"  Correct but flagged      : {false_positives}/{cot_correct_count}  (false positives)")
    print(f"  Errors missed            : {false_negatives}/{cot_errors}  (false negatives)")
    print(f"  Correct and passed       : {true_negatives}/{cot_correct_count}  (true negatives)")
    print()
    print(f"  Precision                : {precision:.1%}")
    print(f"  Recall                   : {recall:.1%}")
    print(f"  F1                       : {f1:.4f}")
    print("=" * 74)

    os.makedirs(output_dir, exist_ok=True)
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
        description="Verify-only benchmark: flag CoT answers using verification, measure detection quality.",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=os.path.join(SCRIPT_DIR, "..", "data", "dataset.json"),
        help="Path to dataset JSON (default: data/dataset.json).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Root output directory. A timestamped sub-folder is created automatically.",
    )
    parser.add_argument(
        "--num-problems", "-n",
        type=int,
        default=None,
        help="Number of problems to run (default: all).",
    )
    parser.add_argument(
        "--client", "-c",
        choices=["openai", "kimi", "deepseek", "vllm"],
        default="openai",
        help="Reasoning client for CoT generation (default: openai). Verification always uses openai.",
    )
    parser.add_argument(
        "--skip-tptp",
        action="store_true",
        help="Skip the TPTP/prover step and only use fact/rule verification.",
    )
    args = parser.parse_args()

    reasoning_client = make_client(args.client)
    verifier_client = OpenAILLMClient()

    run_id = datetime.now().strftime("verify_run_%Y%m%d_%H%M%S")
    base_dir = args.output_dir or os.path.join(SCRIPT_DIR, "..", "benchmark_outputs")
    output_dir = os.path.join(base_dir, run_id)

    run_verify_benchmark(
        dataset_path=args.dataset,
        output_dir=output_dir,
        num_problems=args.num_problems,
        reasoning_client=reasoning_client,
        verifier_client=verifier_client,
        skip_tptp=args.skip_tptp,
    )
