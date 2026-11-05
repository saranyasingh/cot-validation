"""
Graph data model, validation and automatic labels.

A graph has a dictionary (sorts, entities, relations) and two node types:
statements (shape fact, rule or closure) and steps (hyperedges joining premises and a
warrant to one conclusion). Fields marked "computed" are rewritten by `refresh` and
are never authored by agents.
"""

import json
import re
from fractions import Fraction

BUILTIN_SORTS = {"num": "a number", "thing": "any individual"}
SHAPES = {"fact", "rule", "closure"}
KINDS = {"from_problem", "math_principle", "definitional", "commonsense",
         "closure", "normative", "stipulative", "unjustified"}
MOVES = {"apply", "contrapose", "eliminate", "define", "aggregate", "chain"}
ARITH_OPS = {"sum", "difference", "product", "quotient"}
COMPARE_OPS = {"lt", "le", "gt", "ge", "eq"}

SYMBOL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
NODE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?(/\d+)?$")

COLLECTIONS = ("sorts", "entities", "relations", "statements", "steps")


# ─── Values ───────────────────────────────────────────────────────────────────

def is_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, str) and bool(NUMBER_RE.match(value.strip()))


def to_fraction(value) -> Fraction:
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(str(value).strip()) if isinstance(value, str) else Fraction(value)


def ground_value(value):
    """A filler as a hashable value: Fraction for numbers, str for entity ids."""
    return to_fraction(value) if is_number(value) else value


def literals(fact_content: dict) -> list[dict]:
    """The literals of a fact: one, or several for an "any_of" disjunction."""
    if "any_of" in fact_content:
        return fact_content["any_of"]
    return [fact_content]


def atom_key(literal: dict) -> tuple:
    return (literal["relation"], tuple(ground_value(a) for a in literal.get("args", [])))


# ─── Graph ────────────────────────────────────────────────────────────────────

class Graph:
    def __init__(self, question: str = "", cot: str = ""):
        self.question = question
        self.cot = cot
        self.sorts: dict[str, dict] = {}
        self.entities: dict[str, dict] = {}
        self.relations: dict[str, dict] = {}
        self.statements: dict[str, dict] = {}
        self.steps: dict[str, dict] = {}
        self.oplog: list[dict] = []
        self.errors: list[str] = []

    def collection(self, name: str) -> dict:
        if name not in COLLECTIONS:
            raise KeyError(f"unknown collection '{name}'")
        return getattr(self, name)

    def all_sorts(self) -> dict:
        return {**BUILTIN_SORTS, **self.sorts}

    def derived_ids(self) -> set[str]:
        return {s.get("conclusion") for s in self.steps.values() if s.get("conclusion")}

    def concluding_step(self, statement_id: str):
        for step in self.steps.values():
            if step.get("conclusion") == statement_id:
                return step
        return None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "cot": self.cot,
            "sorts": self.sorts,
            "entities": self.entities,
            "relations": self.relations,
            "statements": self.statements,
            "steps": self.steps,
            "errors": self.errors,
        }

    def authored_dict(self) -> dict:
        """The graph without computed fields, as shown to agents."""
        computed_statement = {"label", "english"}
        computed_step = {"binding", "status", "status_detail", "english"}
        return {
            "sorts": self.sorts,
            "entities": self.entities,
            "relations": self.relations,
            "statements": {k: {f: v for f, v in s.items() if f not in computed_statement}
                           for k, s in self.statements.items()},
            "steps": {k: {f: v for f, v in s.items() if f not in computed_step}
                      for k, s in self.steps.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        g = cls(data.get("question", ""), data.get("cot", ""))
        for name in COLLECTIONS:
            setattr(g, name, dict(data.get(name, {})))
        for name in ("statements", "steps"):
            for node_id, node in g.collection(name).items():
                node["id"] = node_id
        g.errors = list(data.get("errors", []))
        return g

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Graph":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ─── Quotes ───────────────────────────────────────────────────────────────────

def _norm_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def quote_found(quote: str, source: str) -> bool:
    q = _norm_text(quote or "")
    return bool(q) and q in _norm_text(source)


# ─── Automatic labels ─────────────────────────────────────────────────────────

def rule_mentions_entities(graph: Graph, content: dict) -> bool:
    variables = set(content.get("vars", {}))
    for lit in content.get("if", []) + content.get("then", []):
        for arg in lit.get("args", []):
            if isinstance(arg, str) and arg not in variables and arg in graph.entities:
                return True
    return False


def apply_auto_labels(graph: Graph):
    """Derived/assumption from edges. Kind is left to the classifier, except two hard rules."""
    derived = graph.derived_ids()
    for sid, st in graph.statements.items():
        if sid in derived:
            st["label"] = "derived"
            st["kind"] = None
            continue
        st["label"] = "assumption"
        if st.get("question_quote") and quote_found(st["question_quote"], graph.question):
            st["kind"] = "from_problem"
        elif (st.get("shape") == "rule" and st.get("kind") not in (None, "unjustified")
              and rule_mentions_entities(graph, st.get("content") or {})):
            st["kind"] = "stipulative"


# ─── Validation ───────────────────────────────────────────────────────────────

def _sort_ok(slot_sort: str, entity_sort: str) -> bool:
    return slot_sort == entity_sort or "thing" in (slot_sort, entity_sort)


def _check_literal(graph: Graph, lit: dict, where: str, variables: dict, errors: list):
    rel_id = lit.get("relation")
    rel = graph.relations.get(rel_id)
    if rel is None:
        errors.append(f"{where}: relation '{rel_id}' is not declared")
        return
    args = lit.get("args", [])
    slots = rel.get("slots", [])
    if len(args) != len(slots):
        errors.append(f"{where}: relation '{rel_id}' has {len(slots)} slots but got {len(args)} args")
        return
    for arg, (slot_name, slot_sort) in zip(args, slots):
        if isinstance(arg, str) and arg in variables:
            var_sort = variables[arg]
            if (slot_sort == "num") != (var_sort == "num") or (
                    slot_sort != "num" and not _sort_ok(slot_sort, var_sort)):
                errors.append(f"{where}: variable '{arg}' has sort '{var_sort}' but slot "
                              f"'{slot_name}' of '{rel_id}' needs '{slot_sort}'")
        elif is_number(arg):
            if slot_sort != "num":
                errors.append(f"{where}: number {arg} in slot '{slot_name}' of '{rel_id}', "
                              f"which needs '{slot_sort}'")
        elif isinstance(arg, str) and arg in graph.entities:
            ent_sort = graph.entities[arg].get("sort")
            if slot_sort == "num" or not _sort_ok(slot_sort, ent_sort):
                errors.append(f"{where}: entity '{arg}' has sort '{ent_sort}' but slot "
                              f"'{slot_name}' of '{rel_id}' needs '{slot_sort}'")
        else:
            errors.append(f"{where}: '{arg}' is not a declared entity, variable or number")


def _check_rule(graph: Graph, sid: str, content: dict, errors: list):
    variables = content.get("vars", {})
    sorts = graph.all_sorts()
    for var, sort in variables.items():
        if sort not in sorts:
            errors.append(f"{sid}: variable '{var}' has undeclared sort '{sort}'")
    if not content.get("then"):
        errors.append(f"{sid}: rule has no 'then' pattern")
    for lit in content.get("if", []) + content.get("then", []):
        _check_literal(graph, lit, sid, variables, errors)
    if content.get("form", "implies") not in ("implies", "iff"):
        errors.append(f"{sid}: rule form must be 'implies' or 'iff'")
    for calc in content.get("calc", []):
        op = calc.get("op")
        if op not in ARITH_OPS | COMPARE_OPS:
            errors.append(f"{sid}: unknown calc op '{op}'")
            continue
        for arg in calc.get("args", []):
            if not is_number(arg) and variables.get(arg) != "num":
                errors.append(f"{sid}: calc argument '{arg}' must be a num variable or a number")
        if op in ARITH_OPS:
            result = calc.get("result")
            if variables.get(result) != "num":
                errors.append(f"{sid}: calc result '{result}' must be a declared num variable")
            if len(calc.get("args", [])) < 2:
                errors.append(f"{sid}: calc '{op}' needs at least two args")
        elif len(calc.get("args", [])) != 2:
            errors.append(f"{sid}: comparison '{op}' needs exactly two args")


def _check_closure(graph: Graph, sid: str, content: dict, errors: list):
    rel = graph.relations.get(content.get("relation"))
    if rel is None:
        errors.append(f"{sid}: closure relation '{content.get('relation')}' is not declared")
        return
    if len(rel.get("slots", [])) != 2:
        errors.append(f"{sid}: closure relation '{content['relation']}' must have exactly two slots "
                      f"(whole, member)")
        return
    _check_literal(graph, {"relation": content["relation"],
                           "args": [content.get("whole"), (content.get("members") or [None])[0]]},
                   sid, {}, errors)
    for member in content.get("members", []):
        if member not in graph.entities:
            errors.append(f"{sid}: closure member '{member}' is not a declared entity")
    if not content.get("members"):
        errors.append(f"{sid}: closure has no members")


def _has_cycle(graph: Graph) -> bool:
    produced_by = {s["conclusion"]: s for s in graph.steps.values() if s.get("conclusion")}
    state: dict[str, int] = {}

    def visit(sid: str) -> bool:
        if state.get(sid) == 1:
            return True
        if state.get(sid) == 2:
            return False
        state[sid] = 1
        step = produced_by.get(sid)
        if step:
            inputs = list(step.get("premises", [])) + ([step["warrant"]] if step.get("warrant") else [])
            if any(visit(i) for i in inputs):
                return True
        state[sid] = 2
        return False

    return any(visit(sid) for sid in graph.statements)


def validate(graph: Graph) -> list[str]:
    errors: list[str] = []
    sorts = graph.all_sorts()

    for name in ("sorts", "entities", "relations"):
        for symbol in graph.collection(name):
            if not SYMBOL_RE.match(symbol):
                errors.append(f"{name} id '{symbol}' must be lowercase snake_case")
    for name in ("statements", "steps"):
        for node_id in graph.collection(name):
            if not NODE_ID_RE.match(node_id):
                errors.append(f"{name} id '{node_id}' must be alphanumeric")
    overlap = set(graph.statements) & set(graph.steps)
    if overlap:
        errors.append(f"ids used for both a statement and a step: {sorted(overlap)}")

    for eid, ent in graph.entities.items():
        if ent.get("sort") not in sorts or ent.get("sort") == "num":
            errors.append(f"entity '{eid}' has invalid sort '{ent.get('sort')}'")
    for rid, rel in graph.relations.items():
        slots = rel.get("slots", [])
        slot_names = []
        for s in slots:
            if isinstance(s, (list, tuple)) and s:
                slot_names.append(s[0])
            elif isinstance(s, dict):
                slot_names.append(s.get("name") or s.get("slot"))
        for slot_name, slot_sort in rel.get("slots", []):
            if slot_sort not in sorts:
                errors.append(f"relation '{rid}': slot '{slot_name}' has undeclared sort '{slot_sort}'")
        sentence = rel.get("sentence", "")
        placeholders = set(re.findall(r"\{(\w+)\}", sentence))
        if placeholders != set(slot_names):
            errors.append(f"relation '{rid}': sentence must use exactly the placeholders "
                          f"{sorted(slot_names)}, found {sorted(placeholders)}")

    derived = graph.derived_ids()
    for sid, st in graph.statements.items():
        shape = st.get("shape")
        content = st.get("content")
        if shape not in SHAPES:
            errors.append(f"{sid}: shape must be one of {sorted(SHAPES)}")
            continue
        if content is None:
            pass
        elif not isinstance(content, dict):
            errors.append(f"{sid}: content must be an object or null")
        elif shape == "fact":
            for lit in literals(content):
                _check_literal(graph, lit, sid, {}, errors)
        elif shape == "rule":
            _check_rule(graph, sid, content, errors)
        else:
            _check_closure(graph, sid, content, errors)

        if st.get("question_quote") and not quote_found(st["question_quote"], graph.question):
            errors.append(f"{sid}: question_quote is not a verbatim substring of the question")
        if st.get("cot_quote") and not quote_found(st["cot_quote"], graph.cot):
            errors.append(f"{sid}: cot_quote is not a verbatim substring of the CoT")
        if sid not in derived:
            if st.get("kind") is not None and st.get("kind") not in KINDS:
                errors.append(f"{sid}: kind must be one of {sorted(KINDS)}")
            if st.get("stated_in_cot") and not st.get("cot_quote"):
                errors.append(f"{sid}: stated_in_cot is true but there is no cot_quote")

    for step_id, step in graph.steps.items():
        if step.get("move") not in MOVES:
            errors.append(f"{step_id}: move must be one of {sorted(MOVES)}")
        for ref in list(step.get("premises", [])) + [step.get("warrant"), step.get("conclusion")]:
            if ref is not None and ref not in graph.statements:
                errors.append(f"{step_id}: references unknown statement '{ref}'")
        if not step.get("conclusion"):
            errors.append(f"{step_id}: has no conclusion")
        if step.get("cot_quote") and not quote_found(step["cot_quote"], graph.cot):
            errors.append(f"{step_id}: cot_quote is not a verbatim substring of the CoT")
        if step.get("move") == "aggregate":
            w = graph.statements.get(step.get("warrant") or "")
            if not w or w.get("shape") != "closure":
                errors.append(f"{step_id}: an aggregate step needs a closure statement as its warrant")

    conclusions = [s.get("conclusion") for s in graph.steps.values() if s.get("conclusion")]
    for sid in {c for c in conclusions if conclusions.count(c) > 1}:
        errors.append(f"{sid}: is the conclusion of more than one step")
    if _has_cycle(graph):
        errors.append("the steps form a cycle")
    return errors
