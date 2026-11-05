"""
Build an inference graph from a chain of thought and write graph.json, oplog.jsonl,
graph.tptp, argument.txt and graph.html.

Usage:
  python -m graph_pipeline.run --item benchmark_outputs/verify_run_20261001_150731/item_0023
  python -m graph_pipeline.run --question-file q.txt --cot-file cot.txt -o out/
  python -m graph_pipeline.run --graph graph_pipeline/examples/item_0023.json   # no LLM
"""

import argparse
import json
import os

from graph_pipeline.build import build_graph, problems
from graph_pipeline.compile_tff import compile_graph
from graph_pipeline.model import Graph
from graph_pipeline.ops import refresh, step_order
from graph_pipeline.render_en import render_argument
from graph_pipeline.visualize import write_html


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def write_outputs(graph: Graph, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    graph.save(os.path.join(out_dir, "graph.json"))
    with open(os.path.join(out_dir, "oplog.jsonl"), "w") as f:
        for op in graph.oplog:
            f.write(json.dumps(op, default=str) + "\n")
    with open(os.path.join(out_dir, "graph.tptp"), "w") as f:
        f.write(compile_graph(graph))
    with open(os.path.join(out_dir, "argument.txt"), "w") as f:
        f.write(render_argument(graph, step_order(graph)) + "\n")
    write_html(graph, os.path.join(out_dir, "graph.html"))


def print_summary(graph: Graph, out_dir: str):
    statuses: dict[str, int] = {}
    for step in graph.steps.values():
        statuses[step.get("status")] = statuses.get(step.get("status"), 0) + 1
    print(f"\n{len(graph.statements)} statements, {len(graph.steps)} steps: "
          + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    for step_id, step in graph.steps.items():
        if step.get("status") == "arithmetic_mismatch":
            print(f"  reasoning error reported in {step_id}: {step.get('status_detail')}")
    remaining = problems(graph)
    if remaining:
        print(f"{len(remaining)} unresolved problems:")
        for p in remaining:
            print(f"  - {p}")
    print(f"Wrote {out_dir}/graph.html")


def main():
    parser = argparse.ArgumentParser(description="Build a living inference graph from a CoT.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--item", help="benchmark item directory with question.txt and cot.txt")
    src.add_argument("--question-file", help="question text file (use with --cot-file)")
    src.add_argument("--graph", help="existing graph JSON: refresh and re-render, no LLM calls")
    parser.add_argument("--cot-file", help="chain-of-thought text file")
    parser.add_argument("--output-dir", "-o", help="output directory")
    parser.add_argument("--client", "-c", default="bedrock",
                        help="LLM client from main_pipeline/clients.py (default: bedrock)")
    parser.add_argument("--repair-rounds", type=int, default=3)
    args = parser.parse_args()

    if args.graph:
        data = json.loads(_read(args.graph))
        if data.get("source_item") and not data.get("question"):
            data["question"] = _read(os.path.join(data["source_item"], "question.txt"))
            data["cot"] = _read(os.path.join(data["source_item"], "cot.txt"))
        graph = Graph.from_dict(data)
        refresh(graph)
        out_dir = args.output_dir or os.path.join(
            "graph_outputs", os.path.splitext(os.path.basename(args.graph))[0])
    else:
        if args.item:
            question = _read(os.path.join(args.item, "question.txt"))
            cot = _read(os.path.join(args.item, "cot.txt"))
            default_out = os.path.join("graph_outputs", os.path.basename(os.path.normpath(args.item)))
        else:
            if not args.cot_file:
                parser.error("--question-file needs --cot-file")
            question, cot = _read(args.question_file), _read(args.cot_file)
            default_out = os.path.join("graph_outputs", "custom")
        from dotenv import load_dotenv
        load_dotenv()
        if args.client == "bedrock":
            if os.getenv("BEDROCK_API_KEY") and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.environ["BEDROCK_API_KEY"]
            os.environ.setdefault("BEDROCK_MODEL", "anthropic.claude-opus-5-5")
            os.environ.setdefault("ANTHROPIC_MAX_TOKENS", "32000")
        from main_pipeline.clients import make_client
        graph = build_graph(question, cot, make_client(args.client), args.repair_rounds)
        out_dir = args.output_dir or default_out

    write_outputs(graph, out_dir)
    print_summary(graph, out_dir)


if __name__ == "__main__":
    main()
