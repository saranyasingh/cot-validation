"""Deterministic graph-to-TFF compilation. Each graph piece has exactly one translation."""

from graph_pipeline.model import Graph, is_number, literals, to_fraction

ARITH_TFF = {"sum": "$sum", "difference": "$difference", "product": "$product",
             "quotient": "$quotient"}
COMPARE_TFF = {"lt": "$less", "le": "$lesseq", "gt": "$greater", "ge": "$greatereq"}


def tff_sort(sort: str) -> str:
    return "$rat" if sort == "num" else sort


def tff_number(value) -> str:
    x = to_fraction(value)
    return f"{x.numerator}/{x.denominator}"


def tff_var(var: str) -> str:
    return "V_" + var


def tff_term(arg, variables=()) -> str:
    if isinstance(arg, str) and arg in variables:
        return tff_var(arg)
    if is_number(arg):
        return tff_number(arg)
    return str(arg)


def tff_literal(lit: dict, variables=()) -> str:
    args = lit.get("args", [])
    atom = lit["relation"] if not args else \
        f"{lit['relation']}({', '.join(tff_term(a, variables) for a in args)})"
    return f"~{atom}" if lit.get("negated") else atom


def _nest(op: str, terms: list[str]) -> str:
    out = terms[0]
    for t in terms[1:]:
        out = f"{op}({out}, {t})"
    return out


def tff_calc(calc: dict, variables) -> str:
    args = [tff_term(a, variables) for a in calc.get("args", [])]
    op = calc["op"]
    if op in ARITH_TFF:
        return f"{tff_var(calc['result'])} = {_nest(ARITH_TFF[op], args)}"
    if op == "eq":
        return f"{args[0]} = {args[1]}"
    return f"{COMPARE_TFF[op]}({args[0]}, {args[1]})"


def _disjunction(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else "(" + " | ".join(parts) + ")"


def _conjunction(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else "(" + " & ".join(parts) + ")"


def statement_formula(graph: Graph, st: dict) -> str:
    content = st.get("content") or {}
    shape = st.get("shape")
    if shape == "fact":
        return _disjunction([tff_literal(lit) for lit in literals(content)])
    if shape == "rule":
        variables = content.get("vars", {})
        conditions = [tff_literal(l, variables) for l in content.get("if", [])]
        conditions += [tff_calc(c, variables) for c in content.get("calc", [])]
        then = _disjunction([tff_literal(l, variables) for l in content.get("then", [])])
        body = then if not conditions else \
            f"({_conjunction(conditions)} {'<=>' if content.get('form') == 'iff' else '=>'} {then})"
        if not variables:
            return body
        binders = ", ".join(f"{tff_var(v)}: {tff_sort(s)}" for v, s in variables.items())
        return f"![{binders}]: {body}"
    if shape == "closure":
        member_sort = graph.relations[content["relation"]]["slots"][1][1]
        member_test = " | ".join(f"V_m = {m}" for m in content["members"])
        return (f"![V_m: {tff_sort(member_sort)}]: "
                f"({content['relation']}({content['whole']}, V_m) <=> ({member_test}))")
    raise ValueError(f"unknown shape {shape}")


def aggregate_axiom(graph: Graph, step: dict) -> str | None:
    """The finite sum axiom for an aggregate step, built from its premises and conclusion."""
    try:
        return _aggregate_axiom(graph, step)
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def _aggregate_axiom(graph: Graph, step: dict) -> str | None:
    premises = [graph.statements[p] for p in step.get("premises", []) if p in graph.statements]
    conclusion = graph.statements.get(step.get("conclusion"))
    if not premises or conclusion is None:
        return None
    binders, conditions, terms = [], [], []
    for i, p in enumerate(premises, 1):
        lit = literals(p["content"])[0]
        num_positions = [j for j, a in enumerate(lit["args"]) if is_number(a)]
        if len(num_positions) != 1:
            return None
        var = f"N{i}"
        args = [var if j == num_positions[0] else tff_term(a) for j, a in enumerate(lit["args"])]
        binders.append(f"{var}: $rat")
        conditions.append(f"{lit['relation']}({', '.join(args)})")
        terms.append(var)
    clit = literals(conclusion["content"])[0]
    num_positions = [j for j, a in enumerate(clit["args"]) if is_number(a)]
    if len(num_positions) != 1:
        return None
    total = _nest("$sum", terms)
    cargs = [total if j == num_positions[0] else tff_term(a) for j, a in enumerate(clit["args"])]
    return (f"![{', '.join(binders)}]: ({_conjunction(conditions)} => "
            f"{clit['relation']}({', '.join(cargs)}))")


def declarations(graph: Graph) -> list[str]:
    lines = []
    used_sorts = {e.get("sort") for e in graph.entities.values()}
    for rel in graph.relations.values():
        used_sorts |= {s for _, s in rel.get("slots", [])}
    for st in graph.statements.values():
        if st.get("shape") == "rule":
            used_sorts |= set((st.get("content") or {}).get("vars", {}).values())
    for sort in sorted(s for s in used_sorts if s and s != "num"):
        lines.append(f"tff({sort}_type, type, {sort}: $tType).")
    for eid, ent in graph.entities.items():
        lines.append(f"tff({eid}_decl, type, {eid}: {tff_sort(ent['sort'])}).")
    for rid, rel in graph.relations.items():
        slots = [tff_sort(s) for _, s in rel.get("slots", [])]
        signature = "$o" if not slots else \
            (slots[0] if len(slots) == 1 else "(" + " * ".join(slots) + ")") + " > $o"
        lines.append(f"tff({rid}_decl, type, {rid}: {signature}).")
    by_sort: dict[str, list[str]] = {}
    for eid, ent in graph.entities.items():
        by_sort.setdefault(ent["sort"], []).append(eid)
    for sort, ids in by_sort.items():
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                lines.append(f"tff({a}_neq_{b}, axiom, {a} != {b}).")
    return lines


def compile_statement(graph: Graph, sid: str, role: str = "axiom") -> str:
    return f"tff({sid.lower()}, {role}, {statement_formula(graph, graph.statements[sid])})."


def compile_step(graph: Graph, step_id: str) -> str:
    """One problem: premises and warrant as axioms, the conclusion as the conjecture."""
    step = graph.steps[step_id]
    lines = [f"% Step {step_id} ({step.get('move')})"] + declarations(graph)
    for ref in list(step.get("premises", [])) + ([step["warrant"]] if step.get("warrant") else []):
        if ref in graph.statements:
            lines.append(compile_statement(graph, ref))
    if step.get("move") == "aggregate":
        axiom = aggregate_axiom(graph, step)
        if axiom:
            lines.append(f"tff({step_id.lower()}_agg, axiom, {axiom}).")
    if step.get("conclusion") in graph.statements:
        lines.append(compile_statement(graph, step["conclusion"], "conjecture"))
    return "\n".join(lines) + "\n"


def compile_graph(graph: Graph) -> str:
    """Whole graph: assumptions as axioms, derived statements as lemmas."""
    derived = graph.derived_ids()
    lines = ["% Declarations"] + declarations(graph) + ["", "% Statements"]
    for sid, st in graph.statements.items():
        try:
            lines.append(compile_statement(graph, sid, "lemma" if sid in derived else "axiom"))
        except (KeyError, IndexError, TypeError, ValueError) as e:
            lines.append(f"% {sid}: could not compile ({e})")
    for step_id, step in graph.steps.items():
        if step.get("move") == "aggregate":
            axiom = aggregate_axiom(graph, step)
            if axiom:
                lines.append(f"tff({step_id.lower()}_agg, axiom, {axiom}).")
    return "\n".join(lines) + "\n"
