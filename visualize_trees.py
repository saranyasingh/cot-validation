#!/usr/bin/env python3
"""
Visualize derivation trees from a benchmark results directory.

Usage:
    python visualize_trees.py <results_dir>
    python visualize_trees.py benchmark_outputs/verify_run_20260422_140012

Outputs:
    <results_dir>/report.html  — full summary page with all items + inline trees
    <results_dir>/fn_trees/    — individual SVGs for false negatives only
"""

import json
import re
import sys
import textwrap
from pathlib import Path

import graphviz

# ── Node visual styles ─────────────────────────────────────────────────────────

FILL = {
    "fact":          "#a8d5a2",  # green
    "rule":          "#a8c8e8",  # blue
    "inference_ok":  "#f5e642",  # yellow
    "inference_err": "#f08080",  # red
}

OUTCOME_BADGE = {
    "TP": ("#1a7a1a", "white"),
    "TN": ("#1a1a7a", "white"),
    "FP": ("#b85c00", "white"),
    "FN": ("#8b0000", "white"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_premises(formula: str) -> list[str]:
    """Extract premise node IDs from 'from X, Y and Z by ...' formula strings."""
    m = re.match(r"from\s+(.+?)\s+by\s+", formula, re.IGNORECASE)
    if not m:
        return []
    return re.findall(r"(FACT-\d+|RULE-\d+|INF-\d+)", m.group(1))


def wrap_label(text: str, width: int = 38) -> str:
    return "\\n".join(textwrap.wrap(text, width))


def build_dot(item_id: str, tree: dict, meta: dict) -> graphviz.Digraph:
    nodes = tree["nodes"]
    root  = tree.get("root", "")
    outcome = meta.get("outcome", "")

    dot = graphviz.Digraph(
        name=item_id,
        graph_attr={
            "rankdir":  "BT",
            "fontname": "Helvetica",
            "fontsize": "10",
            "label":    f"{item_id}  [{outcome}]",
            "labelloc": "t",
        },
        node_attr={"fontname": "Helvetica", "fontsize": "9"},
        edge_attr={"fontname": "Helvetica", "fontsize": "8", "color": "#555555"},
    )

    for nid, node in nodes.items():
        ntype = node["type"]
        is_root = (nid == root)

        if ntype == "fact":
            fill  = FILL["fact"]
            shape = "box"
            label = f"{nid}\\n{wrap_label(node['formula'])}\\n({node.get('verify_status', '')})"
        elif ntype == "rule":
            fill  = FILL["rule"]
            shape = "ellipse"
            label = f"{nid}\\n{wrap_label(node['formula'])}"
        else:
            fill  = FILL["inference_err"] if node.get("error") else FILL["inference_ok"]
            shape = "diamond"
            status = node.get("tptp_status", "")
            reason = node.get("tptp_reason", "") or ""
            label  = f"{nid}\\n{wrap_label(node['gloss'])}\\n{status}"
            if reason:
                label += f"\\n({wrap_label(reason, 30)})"

        dot.node(
            nid, label=label, shape=shape,
            style="filled,bold" if is_root else "filled",
            fillcolor=fill,
            peripheries="2" if is_root else "1",
        )

    for nid, node in nodes.items():
        if node["type"] == "inference":
            for pid in (node.get("premises") or parse_premises(node.get("formula", ""))):
                if pid in nodes:
                    dot.edge(pid, nid)

    return dot


def svg_string(dot: graphviz.Digraph) -> str:
    raw = dot.pipe(format="svg").decode("utf-8")
    # strip XML declaration so it embeds cleanly in HTML
    return re.sub(r"<\?xml[^?]*\?>\s*<!DOCTYPE[^>]*>\s*", "", raw, flags=re.DOTALL)


# ── Log reconstruction ─────────────────────────────────────────────────────────

def build_log_block(summary: dict, results: list[dict]) -> str:
    run_id           = summary.get("run_id", "")
    dataset          = summary.get("dataset", "")
    total            = summary.get("total", 0)
    cot              = summary.get("cot", {})
    ver              = summary.get("verification", {})
    ed               = summary.get("error_detection", {})
    reasoning_client = summary.get("reasoning_client", "openai")
    verifier_client  = summary.get("verifier_client", "openai")

    lines = []
    lines.append(f"=== Verify-Only Benchmark: {dataset} ===")
    lines.append(f"Items: {total} | Run: {run_id}")
    lines.append(f"CoT agent: {reasoning_client} | Verifier: {verifier_client}")
    lines.append("")
    lines.append(f"{'ID':<12} {'Expected':<12} {'CoT Ans':<12} {'CoT OK':<10} {'Flagged':<10} {'Outcome'}")
    lines.append("-" * 74)
    for r in results:
        cot_marker  = "✓" if r.get("cot_correct") else "✗"
        flag_marker = "flagged" if r.get("verify_flagged") else "passed"
        lines.append(
            f"{r['id']:<12} {str(r.get('expected','')):<12} "
            f"{str(r.get('cot_answer','')):<12} {cot_marker:<10} "
            f"{flag_marker:<10} {r.get('outcome','')}"
        )
    lines.append("")
    lines.append("=" * 74)
    lines.append(f"{'RESULTS SUMMARY':^74}")
    lines.append("=" * 74)
    lines.append(f"  Total items              : {total}")
    lines.append(f"  CoT correct              : {cot.get('correct',0)}/{total} ({cot.get('accuracy',0):.1%})")
    lines.append(f"  CoT incorrect            : {cot.get('incorrect',0)}/{total}")
    lines.append("")
    lines.append(f"  -- Verification Error Detection --")
    lines.append(f"  Errors correctly flagged : {ed.get('true_positives',0)}/{cot.get('incorrect',0)}  (true positives)")
    lines.append(f"  Correct but flagged      : {ed.get('false_positives',0)}/{cot.get('correct',0)}  (false positives)")
    lines.append(f"  Errors missed            : {ed.get('false_negatives',0)}/{cot.get('incorrect',0)}  (false negatives)")
    lines.append(f"  Correct and passed       : {ed.get('true_negatives',0)}/{cot.get('correct',0)}  (true negatives)")
    lines.append("")
    lines.append(f"  Precision                : {ed.get('precision',0):.1%}")
    lines.append(f"  Recall                   : {ed.get('recall',0):.1%}")
    lines.append(f"  F1                       : {ed.get('f1',0):.4f}")
    lines.append("=" * 74)
    return "\n".join(lines)


# ── HTML builder ───────────────────────────────────────────────────────────────

OUTCOME_ORDER = ["FN", "FP", "TP", "TN"]

def outcome_badge(outcome: str) -> str:
    bg, fg = OUTCOME_BADGE.get(outcome, ("#888", "white"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 7px;'
            f'border-radius:4px;font-weight:bold;font-size:0.85em">{outcome}</span>')


def item_card(item: dict, svg: str, expanded: bool) -> str:
    item_id = item["id"]
    outcome = item.get("outcome", "")
    question = item.get("question", "")
    expected = item.get("expected", "")
    cot_ans  = item.get("cot_answer", "")
    cot_ok   = item.get("cot_correct", False)
    flagged  = item.get("verify_flagged", False)

    border_color = OUTCOME_BADGE.get(outcome, ("#ccc", "black"))[0]
    open_attr    = "open" if expanded else ""

    meta_html = (
        f'<span style="margin-right:12px"><b>Expected:</b> {expected}</span>'
        f'<span style="margin-right:12px"><b>CoT:</b> {cot_ans} '
        f'{"✓" if cot_ok else "✗"}</span>'
        f'<span><b>Flagged:</b> {"yes" if flagged else "no"}</span>'
    )

    return f"""
<details {open_attr} style="border-left:4px solid {border_color};margin:8px 0;
  padding:0;background:#fafafa;border-radius:4px">
  <summary style="cursor:pointer;padding:10px 14px;list-style:none;
    display:flex;align-items:center;gap:10px;font-family:monospace">
    {outcome_badge(outcome)}
    <b>{item_id}</b>
    <span style="color:#555;font-size:0.9em;flex:1">{question[:120]}{"…" if len(question)>120 else ""}</span>
    <span style="font-size:0.8em;color:#777">{meta_html}</span>
  </summary>
  <div style="padding:10px 14px;border-top:1px solid #e0e0e0">
    <p style="margin:0 0 8px 0;font-family:sans-serif;font-size:0.9em"><b>Question:</b> {question}</p>
    <div style="overflow-x:auto">{svg}</div>
  </div>
</details>"""


def build_html(summary: dict, results: list[dict], svgs: dict[str, str]) -> str:
    run_id           = summary.get("run_id", "")
    dataset          = summary.get("dataset", "")
    reasoning_client = summary.get("reasoning_client", "openai")
    verifier_client  = summary.get("verifier_client", "openai")
    log_text = build_log_block(summary, results)

    sorted_results = sorted(results, key=lambda r: OUTCOME_ORDER.index(r.get("outcome", "TN")))

    cards_html = ""
    for item in sorted_results:
        outcome   = item.get("outcome", "")
        expanded  = (outcome == "FN")
        cards_html += item_card(item, svgs.get(item["id"], ""), expanded)

    ed = summary.get("error_detection", {})
    stat_pills = "".join(
        f'<span style="margin-right:10px">{outcome_badge(k)} {v}</span>'
        for k, v in [
            ("FN", ed.get("false_negatives", 0)),
            ("FP", ed.get("false_positives", 0)),
            ("TP", ed.get("true_positives", 0)),
            ("TN", ed.get("true_negatives", 0)),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Derivation Tree Report — {run_id}</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 20px 32px; background: #f5f5f5; color: #222; }}
  h1   {{ font-size: 1.3em; margin-bottom: 4px; }}
  h2   {{ font-size: 1.05em; color: #444; margin: 24px 0 8px; }}
  pre  {{ background: #1e1e1e; color: #d4d4d4; padding: 16px 20px; border-radius: 6px;
          font-size: 0.82em; line-height: 1.5; overflow-x: auto; white-space: pre; }}
  details > summary::-webkit-details-marker {{ display: none; }}
  details[open] > summary {{ background: #efefef; }}
  svg  {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>Derivation Tree Report</h1>
<p style="color:#555;margin:0 0 4px">{dataset} &nbsp;·&nbsp; {run_id}</p>
<p style="color:#666;font-size:0.88em;margin:0 0 10px">
  CoT agent: <b>{reasoning_client}</b> &nbsp;·&nbsp; Verifier: <b>{verifier_client}</b>
</p>
<p style="margin:0 0 20px">{stat_pills}</p>

<h2>Run Log</h2>
<pre>{log_text}</pre>

<h2>Derivation Trees <span style="font-weight:normal;font-size:0.85em;color:#777">
  (false negatives expanded by default)</span></h2>
{cards_html}
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main(results_dir: Path) -> None:
    results_file = results_dir / "results.json"
    summary_file = results_dir / "summary.json"

    if not results_file.exists():
        sys.exit(f"No results.json in {results_dir}")

    with results_file.open() as f:
        results = json.load(f)
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}

    fn_dir = results_dir / "fn_trees"
    fn_dir.mkdir(exist_ok=True)

    svgs: dict[str, str] = {}
    for item in results:
        item_id = item["id"]
        tree    = item.get("derivation_tree")
        if not tree:
            print(f"  {item_id}: no derivation_tree, skipping")
            continue

        dot = build_dot(item_id, tree, item)
        svgs[item_id] = svg_string(dot)

        if item.get("outcome") == "FN":
            out = fn_dir / item_id
            dot.render(str(out), format="svg", cleanup=True)
            print(f"  {item_id} [FN] -> fn_trees/{item_id}.svg")

    html = build_html(summary, results, svgs)
    report_path = results_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\nReport: {report_path}")
    if any(r.get("outcome") == "FN" for r in results):
        print(f"FN SVGs: {fn_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]))
