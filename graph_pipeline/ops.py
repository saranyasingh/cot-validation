"""
Edits to the living graph. Every change goes through `apply_ops`, is logged in the
graph's oplog, and is followed by `refresh`, which recomputes every computed field.

Ops (JSON):
  {"op": "put",    "collection": "<sorts|entities|relations|statements|steps>", "id": "...", "value": {...}}
  {"op": "delete", "collection": "...", "id": "..."}
"""

import copy
from datetime import datetime

from graph_pipeline.model import COLLECTIONS, Graph, apply_auto_labels, validate
from graph_pipeline.match import match_step
from graph_pipeline.render_en import render_statement, render_step

COMPUTED = {
    "statements": {"label", "english"},
    "steps": {"binding", "status", "status_detail", "english", "prover", "prover_detail"},
}


def apply_ops(graph: Graph, ops: list[dict], agent: str) -> list[str]:
    """Apply ops in order. Returns the rejection reasons for malformed ops."""
    rejected = []
    for op in ops:
        kind, name, node_id = op.get("op"), op.get("collection"), op.get("id")
        if name not in COLLECTIONS or not isinstance(node_id, str) or not node_id:
            rejected.append(f"rejected op {op!r}: needs a valid collection and id")
            continue
        coll = graph.collection(name)
        if kind == "put":
            value = op.get("value")
            if not isinstance(value, dict):
                rejected.append(f"rejected put {name}/{node_id}: value must be an object")
                continue
            value = {k: v for k, v in copy.deepcopy(value).items()
                     if k not in COMPUTED.get(name, set())}
            if name in ("statements", "steps"):
                value["id"] = node_id
            coll[node_id] = value
        elif kind == "delete":
            if node_id not in coll:
                rejected.append(f"rejected delete {name}/{node_id}: no such id")
                continue
            del coll[node_id]
        else:
            rejected.append(f"rejected op {op!r}: op must be 'put' or 'delete'")
            continue
        graph.oplog.append({
            "seq": len(graph.oplog) + 1,
            "time": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            **op,
        })
    refresh(graph)
    return rejected


def collection_ops(name: str, items: dict) -> list[dict]:
    return [{"op": "put", "collection": name, "id": k, "value": v} for k, v in items.items()]


def refresh(graph: Graph):
    """Recompute labels, English, bindings, step statuses and validation errors."""
    try:
        apply_auto_labels(graph)
        errors = validate(graph)
    except Exception as e:  # malformed agent output must surface as an error, not a crash
        errors = [f"validation could not run on malformed content: {e!r}"]
    for st in graph.statements.values():
        try:
            st["english"] = render_statement(graph, st)
        except Exception as e:  # rendering must never break a refresh
            st["english"] = f"[could not render: {e}]"
    for step in graph.steps.values():
        try:
            step["english"] = render_step(graph, step)
        except Exception as e:
            step["english"] = f"[could not render: {e}]"
        result = match_step(graph, step)
        step["status"] = result["status"]
        step["status_detail"] = result.get("detail", "")
        step["binding"] = result.get("binding", {})
        step.pop("prover", None)
        step.pop("prover_detail", None)
    graph.errors = errors


def step_order(graph: Graph) -> list[str]:
    """Steps in dependency order."""
    producer = {s["conclusion"]: sid for sid, s in graph.steps.items() if s.get("conclusion")}
    order, seen = [], set()

    def visit(step_id):
        if step_id in seen:
            return
        seen.add(step_id)
        step = graph.steps[step_id]
        for ref in list(step.get("premises", [])) + [step.get("warrant")]:
            if ref in producer:
                visit(producer[ref])
        order.append(step_id)

    for step_id in graph.steps:
        visit(step_id)
    return order
