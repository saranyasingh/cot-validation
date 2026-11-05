"""
Step matching: compute each step's binding by unifying its rule with its statements.

A rule is grounded by matching each of its if/then patterns against the step's fact
literals (or leaving a pattern unmatched), then evaluating its arithmetic with exact
fractions. The grounded step is propositional, so a truth table decides whether the
premises plus the grounded rule entail the conclusion. This works the same way for
apply, contrapose, eliminate and define.

Statuses:
  matched              the step's pieces fit together; `binding` is set
  gap                  they don't fit; the formalization is incomplete or wrong
  arithmetic_mismatch  they fit except for arithmetic the CoT got wrong (reported only)
  unchecked            the move is not checked in this version (chain)
"""

import itertools
from fractions import Fraction

from graph_pipeline.model import Graph, atom_key, ground_value, is_number, literals
from graph_pipeline.render_en import fmt_number

MAX_COMBINATIONS = 20000


# ─── Propositional formulas ───────────────────────────────────────────────────
# ("atom", key) | ("not", f) | ("and", [f]) | ("or", [f]) | ("implies", a, b)
# | ("iff", a, b) | ("const", bool)

def _lit_formula(key, negated: bool):
    f = ("atom", key)
    return ("not", f) if negated else f


def _fact_formula(content: dict):
    parts = [_lit_formula(atom_key(l), l.get("negated", False)) for l in literals(content)]
    return parts[0] if len(parts) == 1 else ("or", parts)


def _atoms(f, out: set):
    tag = f[0]
    if tag == "atom":
        out.add(f[1])
    elif tag == "not":
        _atoms(f[1], out)
    elif tag in ("and", "or"):
        for g in f[1]:
            _atoms(g, out)
    elif tag in ("implies", "iff"):
        _atoms(f[1], out)
        _atoms(f[2], out)
    return out


def _eval(f, env: dict) -> bool:
    tag = f[0]
    if tag == "atom":
        return env[f[1]]
    if tag == "const":
        return f[1]
    if tag == "not":
        return not _eval(f[1], env)
    if tag == "and":
        return all(_eval(g, env) for g in f[1])
    if tag == "or":
        return any(_eval(g, env) for g in f[1])
    if tag == "implies":
        return (not _eval(f[1], env)) or _eval(f[2], env)
    if tag == "iff":
        return _eval(f[1], env) == _eval(f[2], env)
    raise ValueError(tag)


def entails(premises: list, conclusion) -> bool:
    atoms: set = set()
    for f in premises + [conclusion]:
        _atoms(f, atoms)
    atoms = sorted(atoms, key=repr)
    for values in itertools.product((False, True), repeat=len(atoms)):
        env = dict(zip(atoms, values))
        if all(_eval(p, env) for p in premises) and not _eval(conclusion, env):
            return False
    return True


# ─── Unification ──────────────────────────────────────────────────────────────

def _unify(pattern: dict, key: tuple, binding: dict, variables: dict):
    relation, args = key
    if pattern["relation"] != relation or len(pattern.get("args", [])) != len(args):
        return None
    new = dict(binding)
    for p, g in zip(pattern["args"], args):
        if isinstance(p, str) and p in variables:
            if (variables[p] == "num") != isinstance(g, Fraction):
                return None
            if p in new and new[p] != g:
                return None
            new[p] = g
        elif ground_value(p) != g:
            return None
    return new


def _instantiate(pattern: dict, binding: dict, variables: dict):
    args = []
    for p in pattern.get("args", []):
        if isinstance(p, str) and p in variables:
            if p not in binding:
                return None
            args.append(binding[p])
        else:
            args.append(ground_value(p))
    return (pattern["relation"], tuple(args))


def _value(arg, binding, variables):
    if isinstance(arg, str) and arg in variables:
        return binding.get(arg)
    return ground_value(arg) if is_number(arg) else None


def _compute(op: str, values: list[Fraction]):
    if op == "sum":
        return sum(values, Fraction(0))
    if op == "difference":
        out = values[0]
        for v in values[1:]:
            out -= v
        return out
    if op == "product":
        out = Fraction(1)
        for v in values:
            out *= v
        return out
    if op == "quotient":
        out = values[0]
        for v in values[1:]:
            if v == 0:
                return None
            out /= v
        return out
    return None


COMPARE = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
           "ge": lambda a, b: a >= b, "eq": lambda a, b: a == b}


def _evaluate_calcs(content: dict, binding: dict, variables: dict):
    """Returns (binding, calc formulas, list of failed calc descriptions)."""
    binding = dict(binding)
    formulas, failures = [], []
    for i, calc in enumerate(content.get("calc", [])):
        values = [_value(a, binding, variables) for a in calc.get("args", [])]
        op = calc.get("op")
        if any(v is None for v in values):
            formulas.append(("atom", ("__calc_unknown__", i)))
            continue
        if op in COMPARE:
            ok = COMPARE[op](values[0], values[1])
            formulas.append(("const", ok))
            if not ok:
                failures.append(f"{op}({', '.join(fmt_number(v) for v in values)}) is false")
            continue
        result = _compute(op, values)
        target = calc.get("result")
        if result is None:
            formulas.append(("const", False))
            failures.append(f"{op} of {[fmt_number(v) for v in values]} is undefined")
        elif target in binding:
            ok = binding[target] == result
            formulas.append(("const", ok))
            if not ok:
                failures.append(f"{target} should be {fmt_number(result)} "
                                f"({op} of {', '.join(fmt_number(v) for v in values)}) "
                                f"but the statement says {fmt_number(binding[target])}")
        else:
            binding[target] = result
            formulas.append(("const", True))
    return binding, formulas, failures


def _rule_instance(content: dict, binding: dict, variables: dict, calc_formulas: list):
    fresh = iter(range(10 ** 6))

    def lit(pattern):
        key = _instantiate(pattern, binding, variables)
        if key is None:
            key = ("__unbound__", next(fresh))
        return _lit_formula(key, pattern.get("negated", False))

    conditions = [lit(p) for p in content.get("if", [])] + calc_formulas
    thens = [lit(p) for p in content.get("then", [])]
    then = thens[0] if len(thens) == 1 else ("or", thens)
    if not conditions:
        return then
    tag = "iff" if content.get("form") == "iff" else "implies"
    return (tag, ("and", conditions), then)


def _display_binding(binding: dict) -> dict:
    return {k: (fmt_number(v) if isinstance(v, Fraction) else v) for k, v in binding.items()}


def match_rule_step(graph: Graph, step: dict) -> dict:
    premises = [graph.statements[p] for p in step.get("premises", [])]
    conclusion = graph.statements[step["conclusion"]]
    warrant = graph.statements.get(step.get("warrant") or "")
    for st in premises + [conclusion]:
        if st.get("shape") != "fact":
            return {"status": "gap", "detail": f"{st['id']} must be a fact for move "
                                               f"'{step.get('move')}'"}

    premise_formulas = [_fact_formula(p["content"]) for p in premises]
    conclusion_formula = _fact_formula(conclusion["content"])

    if warrant is None:
        ok = entails(premise_formulas, conclusion_formula)
        return {"status": "matched" if ok else "gap", "binding": {},
                "detail": "" if ok else "the premises alone do not give the conclusion"}
    if warrant.get("shape") == "fact":
        ok = entails(premise_formulas + [_fact_formula(warrant["content"])], conclusion_formula)
        return {"status": "matched" if ok else "gap", "binding": {},
                "detail": "" if ok else "the premises and warrant do not give the conclusion"}
    if warrant.get("shape") != "rule":
        return {"status": "gap", "detail": f"warrant {warrant['id']} must be a rule"}

    content = warrant["content"]
    variables = content.get("vars", {})
    patterns = content.get("if", []) + content.get("then", [])
    step_keys = sorted({atom_key(l) for st in premises + [conclusion]
                        for l in literals(st["content"])}, key=repr)
    options = [[k for k in step_keys if k[0] == p["relation"]] + [None] for p in patterns]

    arithmetic_failure = None
    tried = 0
    for choice in itertools.product(*options):
        tried += 1
        if tried > MAX_COMBINATIONS:
            break
        binding: dict | None = {}
        for pattern, key in zip(patterns, choice):
            if key is not None:
                binding = _unify(pattern, key, binding, variables)
                if binding is None:
                    break
        if binding is None:
            continue
        full, calc_formulas, failures = _evaluate_calcs(content, binding, variables)
        instance = _rule_instance(content, full, variables, calc_formulas)
        if not failures:
            if entails(premise_formulas + [instance], conclusion_formula):
                return {"status": "matched", "binding": _display_binding(full), "detail": ""}
        elif arithmetic_failure is None:
            relaxed = [("const", True) if f[0] == "const" else f for f in calc_formulas]
            relaxed_instance = _rule_instance(content, full, variables, relaxed)
            if entails(premise_formulas + [relaxed_instance], conclusion_formula):
                arithmetic_failure = (full, failures)

    if arithmetic_failure:
        full, failures = arithmetic_failure
        return {"status": "arithmetic_mismatch", "binding": _display_binding(full),
                "detail": "; ".join(failures)}
    used = {p["relation"] for p in patterns}
    missing = [l["relation"] for st in premises + [conclusion] for l in literals(st["content"])
               if l["relation"] not in used]
    detail = (f"rule {warrant['id']} does not connect the premises to the conclusion"
              + (f"; relations not mentioned by the rule: {sorted(set(missing))}" if missing else ""))
    return {"status": "gap", "detail": detail}


def match_aggregate_step(graph: Graph, step: dict) -> dict:
    closure = graph.statements.get(step.get("warrant") or "")
    if not closure or closure.get("shape") != "closure":
        return {"status": "gap", "detail": "aggregate needs a closure statement as its warrant"}
    members = closure["content"].get("members", [])
    premises = [graph.statements[p] for p in step.get("premises", [])]
    conclusion = graph.statements[step["conclusion"]]

    covered, values, relations = [], [], set()
    for p in premises:
        lits = literals(p.get("content", {})) if p.get("shape") == "fact" else []
        if len(lits) != 1 or lits[0].get("negated"):
            return {"status": "gap", "detail": f"{p['id']} must be a single positive fact"}
        args = lits[0]["args"]
        nums = [a for a in args if is_number(a)]
        parts = [a for a in args if a in members]
        if len(nums) != 1 or len(parts) != 1:
            return {"status": "gap", "detail": f"{p['id']} must name exactly one closure member "
                                               f"and exactly one number"}
        relations.add(lits[0]["relation"])
        covered.append(parts[0])
        values.append(ground_value(nums[0]))
    if len(relations) > 1:
        return {"status": "gap", "detail": f"premises use different relations {sorted(relations)}"}
    if sorted(covered) != sorted(members):
        return {"status": "gap", "detail": f"premises cover {sorted(covered)} but the closure lists "
                                           f"{sorted(members)}"}
    clits = literals(conclusion.get("content", {})) if conclusion.get("shape") == "fact" else []
    cnums = [a for a in (clits[0]["args"] if len(clits) == 1 else []) if is_number(a)]
    if len(cnums) != 1:
        return {"status": "gap", "detail": "the conclusion must be a single fact with one number"}
    total = sum(values, Fraction(0))
    binding = {"parts": {m: fmt_number(v) for m, v in zip(covered, values)},
               "total": fmt_number(total)}
    if ground_value(cnums[0]) != total:
        return {"status": "arithmetic_mismatch", "binding": binding,
                "detail": f"the parts sum to {fmt_number(total)} but the conclusion says "
                          f"{fmt_number(ground_value(cnums[0]))}"}
    return {"status": "matched", "binding": binding, "detail": ""}


def match_step(graph: Graph, step: dict) -> dict:
    refs = list(step.get("premises", [])) + [step.get("conclusion")]
    if step.get("warrant"):
        refs.append(step["warrant"])
    if any(r not in graph.statements for r in refs):
        return {"status": "gap", "detail": "the step references a missing statement"}
    empty = [r for r in refs if not isinstance(graph.statements[r].get("content"), dict)]
    if empty:
        return {"status": "gap", "detail": f"no content yet for {', '.join(empty)}"}
    move = step.get("move")
    try:
        if move == "aggregate":
            return match_aggregate_step(graph, step)
        if move == "chain":
            return {"status": "unchecked", "detail": "chain moves are not checked in this version"}
        return match_rule_step(graph, step)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
        return {"status": "gap", "detail": f"could not match: {e!r}"}
