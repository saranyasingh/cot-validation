"""
Build a graph from a chain of thought. Three agents edit one graph in order:

  1. Structurer   CoT only. Writes claim nodes and inference (step) nodes. No kinds.
  2. Classifier   CoT + problem. Marks where each non-derived claim comes from.
  3. Implicit     CoT + graph. Adds unstated claims the reasoning needs, and links them.

A fourth, non-classifying encoder then fills structured content so the graph compiles
to English and TPTP. Every change goes through apply_ops and is logged.
"""

import json
import re

from graph_pipeline.model import Graph
from graph_pipeline.ops import apply_ops, collection_ops

STRUCTURE_PROMPT = '''\
You are turning a chain of thought into an argument graph. Do not classify anything.
Do not look for a problem statement. Use only the CoT below.

The graph has two node types:
- CLAIMS: things the CoT asserts. A claim is a "fact" (a specific statement), a "rule"
  (a general if/then or "any/every" principle), or a "closure" (a list is complete).
- INFERENCES (steps): one inferential move. A step has premises (claim ids), an optional
  warrant (the rule or closure that licenses the move), and exactly one conclusion claim.

Moves. Default is apply. Use aggregate only when the CoT adds several already-computed
part-totals to get a grand total (3 + 4 + 0 = 7). A single product like 3 x 1 = 3 is apply,
not aggregate.
  apply       default: the premises plus a rule give the conclusion
  contrapose  if A then B; not B; so not A
  eliminate   A or B; not A; so B
  define      A means B; A; so B
  aggregate   sum several part-totals to a whole
  chain       if A then B; if B then C; so if A then C

Rules:
- One step per inferential move the CoT actually makes. Do not add, skip or merge moves.
- Each claim node is ONE claim: one assertion, one idea. If a sentence packs two claims
  (a closure and a summing rule, or a fact and an arithmetic principle), split it into
  two nodes. Do not put "and so" / "therefore" conclusions in the same node as their grounds.
- Copy a short verbatim "cot_quote" from the CoT for every claim and step you can locate.
- Do NOT invent missing principles, closure claims, or any other unstated assumption.
  If a step has no warrant in the CoT, set "warrant" to null.
- Do NOT assign a kind, origin, or source. Another agent will classify claims later.
- Plain arithmetic (3 x 1 = 3) is not a claim; put it on the step as "stated_computation".
- Use ids C1, C2, ... for claims and S1, S2, ... for steps.

Return ONLY a JSON object:
{{
  "claims": [
    {{"id": "C1", "shape": "fact", "text": "worded as in the CoT",
      "cot_quote": "verbatim from the CoT"}}
  ],
  "inferences": [
    {{"id": "S1", "move": "apply", "text": "the move in the CoT's words",
      "premises": ["C2", "C3"], "warrant": null, "conclusion": "C4",
      "stated_computation": "3 x 1 = 3" or null,
      "cot_quote": "verbatim from the CoT"}}
  ]
}}

=== CHAIN OF THOUGHT ===
{cot}
'''

CLASSIFY_PROMPT = '''\
You are labeling claims already in an argument graph. Do not add, delete or rewrite claims.
Do not change the steps. Only say where each claim comes from.

A claim that is the conclusion of a step is derived. Skip those; they have no kind.

For every other claim, pick exactly one kind:
  from_problem     the problem statement itself says this (possibly in other words)
  math_principle   a modeling choice such as "total = size times per-person amount"
  definitional     the meaning of a word ("a dozen is 12")
  commonsense      stable world knowledge ("a week has 7 days")
  closure          a list is complete ("these are all the groups")
  normative        a value judgment
  stipulative      a problem-specific rule that names this problem's entities
  unjustified      you cannot tell, or the claim is not supported by the problem or
                   any of the other kinds

If the kind is from_problem, give a short verbatim quote from the problem. Otherwise
question_quote is null.

Return ONLY a JSON object:
{{"classifications": {{
  "C1": {{"kind": "from_problem", "question_quote": "He has 8 customers"}},
  "C2": {{"kind": "math_principle", "question_quote": null}}
}}}}

=== PROBLEM ===
{question}

=== CHAIN OF THOUGHT ===
{cot}

=== CLAIMS TO CLASSIFY ===
{claims}
'''

IMPLICIT_PROMPT = '''\
You are looking at an argument graph built from a chain of thought. Some inferences may be
missing a warrant or a premise the CoT never stated. Add only the implicit claims the
reasoning needs, and link them into the existing steps. Do not rewrite the CoT's claims.
Do not change numbers.

An implicit claim is one the CoT relies on without saying it: a general rule, a closure
("these are all the parts"), or a background fact. Mark each one "stated_in_cot": false
and give it a kind (math_principle, definitional, commonsense, closure, normative,
stipulative, or unjustified).

Each node is ONE claim. Split a bundled sentence into separate nodes. Example: do not write
"the three groups are all the customers, they are disjoint, 3+2+3=8, so the total is the
sum." That is four claims:
  - the listed groups are all of the customers (closure)
  - the groups are disjoint (commonsense or closure)
  - 3 + 2 + 3 = 8 (a derived fact, or a fact plus arithmetic)
  - the total is the sum of the part-totals (math_principle)
Give each its own id and its own kind. Link only the claim that actually warrants the step.

Use new ids I1, I2, ... for implicit claims. To attach a claim to a step, put the whole
updated step (same id) with the new id in premises or warrant.
If you add a rule as a warrant, the step's move should be apply (or contrapose/define/chain).
If you add a closure as a warrant for a sum-of-parts step, the move should be aggregate,
the closure claim's shape must be "closure" (not "fact"), and the step's premises should be
only the part-totals (not the closure itself and not the size of the whole).

If nothing is missing, return {{"ops": []}}.

Edits:
  {{"op": "put", "collection": "statements", "id": "I1", "value": {{...}}}}
  {{"op": "put", "collection": "steps", "id": "S4", "value": {{...}}}}

A statement value: shape, text, kind, stated_in_cot (false), cot_quote (null), content (null).
A step value: move, text, premises, warrant, conclusion, stated_computation, cot_quote.

Return ONLY a JSON object: {{"ops": [...]}}

=== CHAIN OF THOUGHT ===
{cot}

=== GRAPH ===
{graph}

=== NOTES ===
{notes}
'''

ENCODE_PROMPT = '''\
You are encoding an existing argument graph into structured content so it can compile to
logic. Do not add or delete claims or steps. Do not change kinds. Do not add details that
are not already in a claim's text.

=== VOCABULARY ===
- "sorts": kinds of thing, e.g. {{"person": {{"description": "a person"}}}}. "num" and "thing"
  are built in; do not declare them.
- "entities": {{"billy": {{"sort": "person", "name": "Billy"}}}}. Use short display names
  taken from the CoT, not from extra scene-setting.
- "relations": predicates with ordered, sorted slots as [name, sort] pairs, and an English
  sentence that uses every slot as a {{placeholder}}.
  Example: {{"slots": [["owner", "person"], ["count", "num"]],
             "sentence": "{{owner}} has {{count}} customers"}}
- Ids are lowercase snake_case. Keep the vocabulary small. Reuse relations.

=== CONTENT BY SHAPE ===
fact:    {{"relation": "group_size", "args": ["billy", "group1", 3], "negated": false}}
rule:    {{"vars": {{"x": "person", "g": "group", "a": "num", "b": "num", "c": "num"}},
          "if":   [{{"relation": "group_size", "args": ["x", "g", "a"]}}],
          "calc": [{{"op": "product", "args": ["a", "b"], "result": "c"}}],
          "then": [{{"relation": "group_total", "args": ["x", "g", "c"]}}],
          "form": "implies"}}
         calc ops: sum, difference, product, quotient (with "result"); lt, le, gt, ge, eq.
closure: {{"whole": "billy", "relation": "customer_group", "members": ["group1", "group2"]}}
         relation must have exactly two slots: [whole, member].

For each step, the premises and warrant should match the conclusion by pattern:
- apply: the rule's "if" uses the premise relations; "then" uses the conclusion relation.
- aggregate: each premise is the same relation naming one closure member and one number;
  the conclusion is one number, their sum.
Copy numbers exactly as the claims state them, even if the arithmetic looks wrong.

Return ONLY a JSON object:
{{
  "sorts": {{...}}, "entities": {{...}}, "relations": {{...}},
  "statements": {{"C1": {{"content": {{...}}}}, ...}}
}}

=== CHAIN OF THOUGHT ===
{cot}

=== GRAPH ===
{graph}
'''


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def _complete_json(client, prompt: str, retries: int = 2):
    last_error = None
    for _ in range(retries + 1):
        raw = client.complete(prompt)
        try:
            return parse_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            prompt = prompt + f"\n\nYour previous reply was not valid JSON ({e}). Return ONLY JSON."
    raise RuntimeError(f"model did not return valid JSON: {last_error}")


def structure(graph: Graph, client, log=print):
    log("[structurer] claims and inferences from the CoT, no classification...")
    out = _complete_json(client, STRUCTURE_PROMPT.format(cot=graph.cot))
    statements = {}
    for st in out.get("claims") or out.get("statements") or []:
        sid = st.pop("id")
        for drop in ("origin", "kind", "question_quote", "stated_in_cot"):
            st.pop(drop, None)
        st["content"] = None
        st["kind"] = None
        st["stated_in_cot"] = bool(st.get("cot_quote"))
        statements[sid] = st
    steps = {}
    for step in out.get("inferences") or out.get("steps") or []:
        sid = step.pop("id")
        if not step.get("warrant"):
            step["warrant"] = None
        steps[sid] = step
    apply_ops(graph, collection_ops("statements", statements) + collection_ops("steps", steps),
              agent="structurer")
    log(f"[structurer] {len(statements)} claims, {len(steps)} inferences")


def classify(graph: Graph, client, log=print):
    log("[classifier] marking where each claim comes from...")
    derived = graph.derived_ids()
    to_label = []
    for sid, st in graph.statements.items():
        if sid in derived:
            continue
        to_label.append({
            "id": sid,
            "shape": st.get("shape"),
            "text": st.get("text"),
            "cot_quote": st.get("cot_quote"),
        })
    if not to_label:
        log("[classifier] no non-derived claims")
        return
    out = _complete_json(client, CLASSIFY_PROMPT.format(
        question=graph.question, cot=graph.cot,
        claims=json.dumps(to_label, indent=1)))
    ops = []
    for sid, fields in (out.get("classifications") or {}).items():
        if sid not in graph.statements or sid in derived:
            continue
        value = dict(graph.statements[sid])
        value["kind"] = fields.get("kind")
        quote = fields.get("question_quote")
        value["question_quote"] = quote if quote else None
        ops.append({"op": "put", "collection": "statements", "id": sid, "value": value})
    apply_ops(graph, ops, agent="classifier")
    kinds = {}
    for st in graph.statements.values():
        if st.get("label") == "assumption":
            kinds[st.get("kind") or "unlabeled"] = kinds.get(st.get("kind") or "unlabeled", 0) + 1
    log("[classifier] " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda x: str(x[0]))))


def implicit_notes(graph: Graph) -> str:
    lines = []
    for step_id, step in graph.steps.items():
        if not step.get("warrant"):
            lines.append(f"{step_id} has no warrant")
        if step.get("status") == "gap" and step.get("status_detail"):
            lines.append(f"{step_id} gap: {step['status_detail']}")
    unlabeled = [sid for sid, st in graph.statements.items()
                 if st.get("label") == "assumption" and not st.get("kind")]
    if unlabeled:
        lines.append("unclassified claims: " + ", ".join(unlabeled))
    return "\n".join(f"- {line}" for line in lines) or "- no missing warrants noted"


def add_implicit(graph: Graph, client, log=print):
    log("[implicit] adding unstated claims the reasoning needs...")
    out = _complete_json(client, IMPLICIT_PROMPT.format(
        cot=graph.cot,
        graph=json.dumps(graph.authored_dict(), indent=1),
        notes=implicit_notes(graph)))
    ops = []
    for op in out.get("ops") or []:
        if op.get("op") == "put" and op.get("collection") == "statements":
            value = dict(op.get("value") or {})
            value["stated_in_cot"] = False
            value["cot_quote"] = None
            value.setdefault("content", None)
            if not value.get("kind"):
                value["kind"] = "unjustified"
            if value.get("kind") == "closure":
                value["shape"] = "closure"
            op = {**op, "value": value}
        ops.append(op)
    rejected = apply_ops(graph, ops, agent="implicit")
    for r in rejected:
        log(f"    ! {r}")
    added = [op["id"] for op in ops
             if op.get("op") == "put" and op.get("collection") == "statements"
             and op["id"].startswith("I")]
    log(f"[implicit] added {len(added)} implicit claims" + (f": {', '.join(added)}" if added else ""))


def _normalize_slots(slots):
    out = []
    for slot in slots or []:
        if isinstance(slot, (list, tuple)) and len(slot) >= 2:
            out.append([slot[0], slot[1]])
        elif isinstance(slot, dict):
            out.append([slot.get("name") or slot.get("slot"), slot.get("sort") or slot.get("type")])
        else:
            out.append(slot)
    return out


def encode(graph: Graph, client, log=print):
    log("[encoder] filling structured content...")
    out = _complete_json(client, ENCODE_PROMPT.format(
        cot=graph.cot, graph=json.dumps(graph.authored_dict(), indent=1)))
    ops = []
    for name in ("sorts", "entities"):
        ops += collection_ops(name, out.get(name, {}))
    relations = {}
    for rid, rel in (out.get("relations") or {}).items():
        rel = dict(rel)
        rel["slots"] = _normalize_slots(rel.get("slots"))
        relations[rid] = rel
    ops += collection_ops("relations", relations)
    for sid, fields in (out.get("statements") or {}).items():
        if sid not in graph.statements or "content" not in fields:
            continue
        value = {**graph.statements[sid], "content": fields["content"]}
        ops.append({"op": "put", "collection": "statements", "id": sid, "value": value})
    apply_ops(graph, ops, agent="encoder")
    log(f"[encoder] {len(graph.relations)} relations, {len(graph.entities)} entities")


def problems(graph: Graph) -> list[str]:
    out = list(graph.errors)
    for step_id, step in graph.steps.items():
        if step.get("status") == "gap":
            out.append(f"{step_id} is a gap: {step.get('status_detail')}")
    return out


def build_graph(question: str, cot: str, client, repair_rounds: int = 3, log=print) -> Graph:
    graph = Graph(question, cot)
    structure(graph, client, log)
    classify(graph, client, log)
    add_implicit(graph, client, log)
    encode(graph, client, log)
    return graph
