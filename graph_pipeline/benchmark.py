"""
Graph benchmark: for each dataset item, a reasoning model writes a CoT, the graph agents
(checking model) turn it into an inference graph, and Vampire proves every step. The
answer is marked verified only if all three checks pass:

  - every step is proved by Vampire
  - no assumption is unjustified or unclassified
  - a final conclusion of the graph states the CoT's answer

The verdict is then compared with whether the CoT's answer is actually correct.

Usage:
  python -m graph_pipeline.benchmark --dataset data/dataset.json -n 20 --client vllm --verifier bedrock
"""

import argparse
import html
import json
import os
import sys
import traceback
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# main_pipeline/main.py imports its siblings by bare name (import verify, import tptp)
sys.path.insert(0, os.path.join(REPO_ROOT, "main_pipeline"))

from graph_pipeline.build import build_graph
from graph_pipeline.model import Graph, is_number, literals, to_fraction
from graph_pipeline.prove import PROVER_TIMEOUT_S, prove_graph, prover_verdict
from graph_pipeline.run import make_graph_client, write_outputs
from main_pipeline.arithmetic import find_vampire
from main_pipeline.benchmark import answers_match, normalize
from main_pipeline.clients import (
    call_log_mark, print_cost_summary, set_cost_context, start_cost_log, summarize_costs,
)
from main_pipeline.main import COT_PROMPT, extract_answer

CLIENTS = ["openai", "claude", "kimi", "deepseek", "vllm", "bedrock"]


def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ─── Checks ───────────────────────────────────────────────────────────────────

def final_conclusions(graph: Graph) -> list[str]:
    """Derived statements that no step uses as a premise or warrant."""
    used = {ref for step in graph.steps.values()
            for ref in list(step.get("premises", [])) + [step.get("warrant")]}
    return [sid for sid in graph.derived_ids() if sid in graph.statements and sid not in used]


def reaches_answer(graph: Graph, answer: str) -> bool:
    try:
        target = to_fraction(normalize(answer))
    except (ValueError, ZeroDivisionError):
        return False
    for sid in final_conclusions(graph):
        st = graph.statements[sid]
        if st.get("shape") != "fact" or not st.get("content"):
            continue
        for lit in literals(st["content"]):
            if any(is_number(a) and to_fraction(a) == target for a in lit.get("args", [])):
                return True
    return False


def unjustified_assumptions(graph: Graph) -> list[str]:
    return [sid for sid, st in graph.statements.items()
            if st.get("label") == "assumption" and st.get("kind") in (None, "unjustified")]


def check_graph(graph: Graph, answer: str) -> dict:
    step_statuses: dict[str, int] = {}
    for step in graph.steps.values():
        step_statuses[step.get("prover")] = step_statuses.get(step.get("prover"), 0) + 1
    unjustified = unjustified_assumptions(graph)
    checks = {
        "prover_verdict": prover_verdict(graph),
        "all_steps_proved": prover_verdict(graph) == "VERIFIED",
        "no_unjustified_assumptions": not unjustified,
        "reaches_answer": reaches_answer(graph, answer),
    }
    reasons = []
    if not checks["all_steps_proved"]:
        reasons.append("not every step was proved")
    if unjustified:
        reasons.append(f"unjustified assumptions: {', '.join(unjustified)}")
    if not checks["reaches_answer"]:
        reasons.append(f"no final conclusion states the answer {answer!r}")
    return {
        **checks,
        "verified": not reasons,
        "reasons": reasons,
        "unjustified": unjustified,
        "num_statements": len(graph.statements),
        "num_steps": len(graph.steps),
        "step_prover_statuses": step_statuses,
        "step_match_statuses": {s: sum(1 for st in graph.steps.values() if st.get("status") == s)
                                for s in {st.get("status") for st in graph.steps.values()}},
        "validation_errors": len(graph.errors),
    }


# ─── Single item ──────────────────────────────────────────────────────────────

def run_item(item: dict, item_dir: str, reasoning_client, verifier_client, prover_timeout: int) -> dict:
    question = item["question"]
    cot = reasoning_client.complete(COT_PROMPT.format(question=question))
    answer = extract_answer(cot)
    _write(os.path.join(item_dir, "question.txt"), question)
    _write(os.path.join(item_dir, "cot.txt"), cot)

    try:
        graph = build_graph(question, cot, verifier_client)
        vampire = prove_graph(graph, prover_timeout)
        write_outputs(graph, item_dir, vampire)
        checks = check_graph(graph, answer)
    except Exception as e:  # one bad item must not end a long cluster run
        _write(os.path.join(item_dir, "error.txt"), traceback.format_exc())
        checks = {"verified": False, "reasons": [f"graph pipeline failed: {e!r}"], "error": repr(e)}
    return {"cot_answer": answer, **checks}


# ─── Report ───────────────────────────────────────────────────────────────────

def write_index(results: list[dict], summary: dict, path: str):
    rows = []
    for r in results:
        link = f'<a href="{html.escape(r["id"])}/graph.html">graph</a>' if not r.get("error") \
            else f'<a href="{html.escape(r["id"])}/error.txt">error</a>'
        rows.append(
            f"<tr class='{r['outcome']}'><td>{html.escape(r['id'])}</td>"
            f"<td>{html.escape(str(r['expected']))}</td><td>{html.escape(r['cot_answer'])}</td>"
            f"<td>{'yes' if r['cot_correct'] else 'no'}</td>"
            f"<td>{'yes' if r['verified'] else 'no'}</td><td>{r['outcome']}</td>"
            f"<td>{html.escape('; '.join(r['reasons']))}</td><td>{link}</td></tr>")
    j = summary["judgment"]
    _write(path, f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Graph benchmark</title>
<style>body{{font-family:-apple-system,sans-serif;font-size:13px;margin:20px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:4px 8px;text-align:left}}
.TP,.TN{{background:#eef8ee}}.FP,.FN{{background:#fbeaea}}</style></head><body>
<h2>Graph benchmark: {html.escape(summary['dataset'])}</h2>
<p>Reasoning: {html.escape(summary['reasoning_model'])} | Checking: {html.escape(summary['verifier_model'])}</p>
<p>CoT accuracy {summary['cot']['correct']}/{summary['total']} |
correct &amp; verified {j['correct_verified']} | correct &amp; not verified {j['correct_not_verified']} |
incorrect &amp; not verified {j['incorrect_not_verified']} | incorrect &amp; verified {j['incorrect_verified']}</p>
<table><tr><th>id</th><th>expected</th><th>CoT answer</th><th>correct</th><th>verified</th>
<th>outcome</th><th>why not verified</th><th></th></tr>{''.join(rows)}</table></body></html>
""")


def summarize(results: list[dict], dataset_name: str, output_dir: str, reasoning_client,
              verifier_client, args) -> dict:
    total = len(results)

    def count(pred):
        return sum(1 for r in results if pred(r))

    correct = count(lambda r: r["cot_correct"])
    tp = count(lambda r: not r["cot_correct"] and not r["verified"])
    fp = count(lambda r: r["cot_correct"] and not r["verified"])
    fn = count(lambda r: not r["cot_correct"] and r["verified"])
    tn = count(lambda r: r["cot_correct"] and r["verified"])
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    step_statuses: dict[str, int] = {}
    for r in results:
        for k, v in (r.get("step_prover_statuses") or {}).items():
            step_statuses[str(k)] = step_statuses.get(str(k), 0) + v
    return {
        "dataset": dataset_name,
        "run_id": os.path.basename(output_dir),
        "total": total,
        "reasoning_client": args.client,
        "verifier_client": args.verifier,
        "reasoning_model": reasoning_client.model,
        "verifier_model": verifier_client.model,
        "prover_timeout_s": args.prover_timeout,
        "cot": {"correct": correct, "incorrect": total - correct,
                "accuracy": round(correct / total, 4) if total else 0},
        "verification": {
            "verified": count(lambda r: r["verified"]),
            "not_verified": count(lambda r: not r["verified"]),
            "pipeline_errors": count(lambda r: r.get("error")),
            "not_all_steps_proved": count(lambda r: r.get("all_steps_proved") is False),
            "has_unjustified_assumptions": count(lambda r: r.get("no_unjustified_assumptions") is False),
            "does_not_reach_answer": count(lambda r: r.get("reaches_answer") is False),
            "step_prover_statuses": step_statuses,
        },
        "judgment": {
            "incorrect_not_verified": tp,
            "correct_verified": tn,
            "incorrect_verified": fn,
            "correct_not_verified": fp,
            "accuracy": round((tp + tn) / total, 4) if total else 0,
        },
        "error_detection": {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
                            "true_negatives": tn, "precision": round(precision, 4),
                            "recall": round(recall, 4), "f1": round(f1, 4)},
    }


def print_summary(summary: dict, results: list[dict]):
    def pct(n, d):
        return f"{n/d:.1%}" if d else "n/a"

    total, c, v, j = summary["total"], summary["cot"], summary["verification"], summary["judgment"]
    e = summary["error_detection"]
    print("\n" + "=" * 74)
    print(f"{'GRAPH BENCHMARK SUMMARY':^74}")
    print("=" * 74)
    print(f"  CoT correct                : {c['correct']}/{total}  ({pct(c['correct'], total)})")
    print(f"  Verified                   : {v['verified']}/{total}  ({pct(v['verified'], total)})")
    print()
    print("  -- Verified vs. correct --")
    print(f"  Correct, verified          : {j['correct_verified']}/{c['correct']}")
    print(f"  Correct, not verified      : {j['correct_not_verified']}/{c['correct']}")
    print(f"  Incorrect, not verified    : {j['incorrect_not_verified']}/{c['incorrect']}  (errors caught)")
    print(f"  Incorrect, verified        : {j['incorrect_verified']}/{c['incorrect']}  (errors missed)")
    print(f"  Judged correctly           : {pct(j['correct_verified'] + j['incorrect_not_verified'], total)}")
    print(f"  Precision / recall / F1    : {e['precision']:.1%} / {e['recall']:.1%} / {e['f1']:.4f}")
    print()
    print("  -- Why answers were not verified (an item can fail several) --")
    print(f"  Not every step proved      : {v['not_all_steps_proved']}")
    print(f"  Unjustified assumptions    : {v['has_unjustified_assumptions']}")
    print(f"  Graph does not reach answer: {v['does_not_reach_answer']}")
    print(f"  Pipeline errors            : {v['pipeline_errors']}")
    print(f"  Step prover statuses       : {v['step_prover_statuses']}")
    missed = [r for r in results if not r["cot_correct"] and r["verified"]]
    if missed:
        print("\n  -- Incorrect answers that were verified --")
        for r in missed:
            print(f"    {r['id']}: expected {r['expected']!r}, CoT said {r['cot_answer']!r}")
    print()
    print_cost_summary(summary["cost"])
    print("=" * 74)


# ─── Runner ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph pipeline benchmark: verified vs. correct.")
    parser.add_argument("--dataset", "-d", default=os.path.join(REPO_ROOT, "data", "dataset.json"))
    parser.add_argument("--num-problems", "-n", type=int, default=None, help="default: all")
    parser.add_argument("--start", type=int, default=0, help="index of the first item to run")
    parser.add_argument("--client", "-c", choices=CLIENTS, default="vllm",
                        help="reasoning client that writes the CoT (default: vllm)")
    parser.add_argument("--verifier", choices=CLIENTS, default="bedrock",
                        help="client for the graph agents (default: bedrock, Claude Opus)")
    parser.add_argument("--prover-timeout", type=int, default=PROVER_TIMEOUT_S,
                        help=f"Vampire seconds per step (default: {PROVER_TIMEOUT_S})")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="root output directory; a timestamped run folder is created inside")
    args = parser.parse_args()

    vampire = find_vampire()
    if vampire is None:
        sys.exit("Vampire not found. Set VAMPIRE_PATH, put it on PATH, or place it in tools/.")

    with open(args.dataset) as f:
        dataset = json.load(f)
    items = dataset["items"][args.start:]
    if args.num_problems is not None:
        items = items[:args.num_problems]
    dataset_name = dataset.get("name", args.dataset)

    reasoning_client = make_graph_client(args.client)
    verifier_client = make_graph_client(args.verifier)

    run_id = datetime.now().strftime("graph_run_%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir or os.path.join(REPO_ROOT, "benchmark_outputs"), run_id)
    os.makedirs(output_dir, exist_ok=True)
    start_cost_log(os.path.join(output_dir, "api_calls.jsonl"))
    run_mark = call_log_mark()

    print(f"=== Graph Benchmark: {dataset_name} ===")
    print(f"Items: {len(items)} | Reasoning: {args.client} ({reasoning_client.model}) | "
          f"Checking: {args.verifier} ({verifier_client.model})")
    print(f"Vampire: {vampire} | {args.prover_timeout}s per step")
    print(f"Outputs: {output_dir}\n")

    results = []
    for n, item in enumerate(items, 1):
        item_id = item["id"]
        print(f"\n[{n}/{len(items)}] {item_id}")
        set_cost_context(item_id=item_id)
        item_mark = call_log_mark()
        try:
            out = run_item(item, os.path.join(output_dir, item_id), reasoning_client,
                           verifier_client, args.prover_timeout)
        except Exception as e:  # e.g. the reasoning model's server went away
            traceback.print_exc()
            out = {"cot_answer": "", "verified": False, "error": repr(e),
                   "reasons": [f"CoT generation failed: {e!r}"]}
        cot_ok = answers_match(out["cot_answer"], str(item["answer"]))
        outcome = {(False, False): "TP", (True, False): "FP",
                   (False, True): "FN", (True, True): "TN"}[(cot_ok, out["verified"])]
        entry = {"id": item_id, "question": item["question"], "expected": item["answer"],
                 "cot_correct": cot_ok, "outcome": outcome, **out,
                 "cost": summarize_costs(item_mark)}
        results.append(entry)
        _write(os.path.join(output_dir, "results.json"), json.dumps(results, indent=2, default=str))
        print(f"  expected {item['answer']!r} | CoT {out['cot_answer']!r} "
              f"({'correct' if cot_ok else 'incorrect'}) | "
              f"{'VERIFIED' if out['verified'] else 'NOT VERIFIED'}"
              + (f" ({'; '.join(out['reasons'])})" if out["reasons"] else "")
              + f" | ${entry['cost']['total_cost_usd']:.4f}")

    set_cost_context()
    summary = summarize(results, dataset_name, output_dir, reasoning_client, verifier_client, args)
    summary["cost"] = summarize_costs(run_mark)
    print_summary(summary, results)
    _write(os.path.join(output_dir, "summary.json"), json.dumps(summary, indent=2))
    write_index(results, summary, os.path.join(output_dir, "index.html"))
    print(f"\nWrote {output_dir}/summary.json, results.json and index.html")


if __name__ == "__main__":
    main()
