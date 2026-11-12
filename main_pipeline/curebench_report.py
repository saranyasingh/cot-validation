"""
Success rates for a CureBench run folder written by curebench_benchmark.py, overall and by
question type. Prints a table and writes <run_dir>/success_rates.json.

Usage:
    python main_pipeline/curebench_report.py benchmark_outputs/curebench_verify_run_<timestamp>
"""

import argparse
import json
import os
import re
from collections import Counter

_VERDICT_RE = re.compile(r'^(FACT|RULE)-\d+:\s*(SUPPORTED|UNSUPPORTED|PLAUSIBLE|IMPLAUSIBLE)\s*[—–-]+\s*(.*)$')


def _rate(n, d):
    return round(n / d, 4) if d else None


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def _verify_mode_rates(rows):
    n = len(rows)
    outcomes = Counter(r["outcome"] for r in rows)
    tp, fp, fn, tn = outcomes["TP"], outcomes["FP"], outcomes["FN"], outcomes["TN"]
    return {
        "items": n,
        "cot_accuracy": _rate(sum(r["cot_correct"] for r in rows), n),
        "verified_rate": _rate(tn + fn, n),
        "accuracy_of_verified_answers": _rate(tn, tn + fn),
        "wrong_answers_caught": _rate(tp, tp + fn),
        "right_answers_rejected": _rate(fp, fp + tn),
        "judged_correctly": _rate(tp + tn, n),
        "failed_fact_rule_check": _rate(sum(not r["passed_verify"] for r in rows), n),
        "failed_tptp_check": _rate(sum(r["passed_tptp"] is False and r["passed_verify"] for r in rows), n),
    }


def _repair_mode_rates(rows):
    n = len(rows)
    verified = [r for r in rows if r["passed_verify"] and r["passed_tptp"]]
    return {
        "items": n,
        "first_cot_accuracy": _rate(sum(r["first_cot_correct"] for r in rows), n),
        "final_cot_accuracy": _rate(sum(r["final_cot_answer"] == r["expected"] for r in rows), n),
        "pipeline_accuracy": _rate(sum(r["pipeline_correct"] for r in rows), n),
        "verified_rate": _rate(len(verified), n),
        "accuracy_of_verified_answers": _rate(sum(r["pipeline_correct"] for r in verified), len(verified)),
        "fixed_by_repair": sum(not r["first_cot_correct"] and r["final_cot_answer"] == r["expected"] for r in rows),
        "broken_by_repair": sum(r["first_cot_correct"] and r["final_cot_answer"] != r["expected"] for r in rows),
        "avg_attempts": round(sum(r["attempts"] for r in rows) / n, 2) if n else None,
    }


def _final_verify_report(item_dir):
    for name in ("verify_report.txt", "verify_final.txt"):
        path = os.path.join(item_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
    return ""


def _verdict_counts(run_dir, ids):
    """How the fact/rule check judged each premise, from the last verify report of each item."""
    counts = Counter()
    for item_id in ids:
        for line in _final_verify_report(os.path.join(run_dir, item_id)).splitlines():
            m = _VERDICT_RE.match(line.strip())
            if not m:
                continue
            kind, status, reason = m.groups()
            if reason.startswith("UNVERIFIED"):
                counts[f"{kind} unverified"] += 1
            elif reason.startswith("CONTRADICTED") or status in ("UNSUPPORTED", "IMPLAUSIBLE"):
                counts[f"{kind} contradicted/implausible"] += 1
            elif "tool evidence" in reason:
                counts[f"{kind} confirmed by tool"] += 1
            else:
                counts[f"{kind} from question/logic"] += 1
    return dict(sorted(counts.items()))


def _tool_usage(run_dir, ids):
    calls = tool_calls = empty = 0
    for item_id in ids:
        path = os.path.join(run_dir, item_id, "charmchat_calls.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                calls += 1
                tool_calls += len(record.get("tool_calls", []))
                store = (record.get("provenance") or {}).get("evidenceStore", {})
                empty += sum(1 for e in store.values() if e.get("isEmpty"))
    return {
        "agent_calls": calls,
        "tool_calls": tool_calls,
        "tool_calls_with_no_data": empty,
        "tool_calls_per_item": round(tool_calls / len(ids), 2) if ids else None,
    }


def build_report(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(run_dir, "results.json")) as f:
        results = json.load(f)
    with open(os.path.join(run_dir, "dataset.json")) as f:
        types = {item["id"]: item.get("question_type", "unknown") for item in json.load(f)["items"]}

    mode = summary.get("mode", "verify")
    rates = _repair_mode_rates if mode == "repair" else _verify_mode_rates
    ids = [r["id"] for r in results]
    by_type = {}
    for qtype in sorted({types.get(i, "unknown") for i in ids}):
        by_type[qtype] = rates([r for r in results if types.get(r["id"], "unknown") == qtype])

    return {
        "run_dir": os.path.abspath(run_dir),
        "mode": mode,
        "reasoning_model": summary.get("reasoning_model"),
        "verifier_model": summary.get("verifier_model"),
        "overall": rates(results),
        "by_question_type": by_type,
        "premise_verdicts": _verdict_counts(run_dir, ids),
        "tool_usage": _tool_usage(run_dir, ids),
        "cost_usd": (summary.get("cost") or {}).get("total_cost_usd"),
    }


def print_report(report: dict):
    overall = report["overall"]
    keys = [k for k in overall if k != "items"]
    types = list(report["by_question_type"])
    width = max(len(k) for k in keys) + 2

    print("=" * 78)
    print(f"CureBench success rates ({report['mode']} mode)")
    print(f"  Reasoning : {report['reasoning_model']}")
    print(f"  Verifier  : {report['verifier_model']}")
    print("=" * 78)
    header = f"{'':<{width}}{'overall':>12}" + "".join(f"{t[:22]:>24}" for t in types)
    print(header)
    print(f"{'items':<{width}}{overall['items']:>12}"
          + "".join(f"{report['by_question_type'][t]['items']:>24}" for t in types))
    for k in keys:
        cells = [overall[k]] + [report["by_question_type"][t][k] for t in types]
        fmt = [(_pct(c) if isinstance(c, float) or c is None else str(c)) for c in cells]
        if k in ("avg_attempts",):
            fmt = [str(c) for c in cells]
        print(f"{k:<{width}}{fmt[0]:>12}" + "".join(f"{c:>24}" for c in fmt[1:]))

    print("\nPremise verdicts (last verification of each item):")
    for k, v in report["premise_verdicts"].items():
        print(f"  {k:<36} {v}")
    print("\nCharmChat tool usage:")
    for k, v in report["tool_usage"].items():
        print(f"  {k:<36} {v}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Success rates for a CureBench run folder.")
    parser.add_argument("run_dir")
    args = parser.parse_args()

    report = build_report(args.run_dir)
    print_report(report)
    out = os.path.join(args.run_dir, "success_rates.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")
