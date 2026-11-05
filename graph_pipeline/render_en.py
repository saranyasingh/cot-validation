"""Deterministic graph-to-English rendering from relation sentence forms."""

from fractions import Fraction

from graph_pipeline.model import Graph, is_number, literals, to_fraction

OP_SYMBOL = {"sum": " + ", "difference": " - ", "product": " × ", "quotient": " ÷ "}
COMPARE_WORDS = {"lt": "is less than", "le": "is at most", "gt": "is greater than",
                 "ge": "is at least", "eq": "equals"}


def fmt_number(value) -> str:
    x = to_fraction(value)
    if x.denominator == 1:
        return str(x.numerator)
    as_float = float(x)
    if Fraction(str(round(as_float, 6))) == x:
        return f"{as_float:g}"
    return f"{x.numerator}/{x.denominator}"


def fmt_arg(graph: Graph, arg, variables=()) -> str:
    if isinstance(arg, str) and arg in variables:
        return arg
    if is_number(arg):
        return fmt_number(arg)
    entity = graph.entities.get(arg)
    return entity.get("name", arg) if entity else str(arg)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def render_literal(graph: Graph, lit: dict, variables=()) -> str:
    rel = graph.relations.get(lit.get("relation"))
    if rel is None:
        return f"[unknown relation {lit.get('relation')}]"
    fill = {slot_name: fmt_arg(graph, arg, variables)
            for (slot_name, _), arg in zip(rel.get("slots", []), lit.get("args", []))}
    try:
        if lit.get("negated"):
            if rel.get("negated_sentence"):
                return rel["negated_sentence"].format(**fill)
            return "it is not the case that " + rel["sentence"].format(**fill)
        return rel["sentence"].format(**fill)
    except (KeyError, IndexError):
        return f"[{lit.get('relation')}({', '.join(map(str, lit.get('args', [])))})]"


def render_fact(graph: Graph, content: dict) -> str:
    parts = [render_literal(graph, lit) for lit in literals(content)]
    if len(parts) == 1:
        return parts[0]
    return "either " + ", or ".join(parts)


def render_calc(graph: Graph, calc: dict, variables) -> str:
    args = [fmt_arg(graph, a, variables) for a in calc.get("args", [])]
    op = calc.get("op")
    if op in OP_SYMBOL:
        return f"{calc.get('result')} = {OP_SYMBOL[op].join(args)}"
    if op in COMPARE_WORDS and len(args) == 2:
        return f"{args[0]} {COMPARE_WORDS[op]} {args[1]}"
    return f"[{op}({', '.join(args)})]"


def render_rule(graph: Graph, content: dict) -> str:
    variables = content.get("vars", {})
    by_sort: dict[str, list[str]] = {}
    for var, sort in variables.items():
        by_sort.setdefault(sort, []).append(var)
    quantifier = ", ".join(
        f"any {'number' if sort == 'num' else sort} {' and '.join(vs)}" if len(vs) == 1
        else f"any {'numbers' if sort == 'num' else sort + 's'} {', '.join(vs)}"
        for sort, vs in by_sort.items())

    conditions = [render_literal(graph, lit, variables) for lit in content.get("if", [])]
    conditions += [render_calc(graph, c, variables) for c in content.get("calc", [])]
    thens = [render_literal(graph, lit, variables) for lit in content.get("then", [])]
    then_text = thens[0] if len(thens) == 1 else "either " + ", or ".join(thens)

    prefix = f"for {quantifier}: " if quantifier else ""
    if not conditions:
        return f"{prefix}{then_text}"
    if content.get("form") == "iff":
        return f"{prefix}{then_text} if and only if {' and '.join(conditions)}"
    return f"{prefix}if {' and '.join(conditions)}, then {then_text}"


def render_closure(graph: Graph, content: dict) -> str:
    rel = graph.relations.get(content.get("relation"))
    members = [fmt_arg(graph, m) for m in content.get("members", [])]
    member_text = members[0] if len(members) == 1 else ", ".join(members[:-1]) + " and " + members[-1]
    if rel is None or len(rel.get("slots", [])) != 2:
        return f"the members are exactly {member_text}, and nothing else"
    whole_slot, member_slot = rel["slots"][0][0], rel["slots"][1][0]
    pattern = rel["sentence"].format(**{whole_slot: fmt_arg(graph, content.get("whole")),
                                        member_slot: "X"})
    return f"the only X such that \"{pattern}\" are {member_text}; there are no others"


def render_clause(graph: Graph, st: dict) -> str:
    """A statement as a lowercase clause with no final period, for use inside sentences."""
    content = st.get("content") or {}
    shape = st.get("shape")
    if shape == "fact":
        return render_fact(graph, content)
    if shape == "rule":
        return render_rule(graph, content)
    if shape == "closure":
        return render_closure(graph, content)
    return "[no content]"


def render_statement(graph: Graph, st: dict) -> str:
    return _capitalize(render_clause(graph, st)) + "."


def render_step(graph: Graph, step: dict) -> str:
    def sentence(sid):
        st = graph.statements.get(sid)
        return render_clause(graph, st) if st else f"[{sid}]"

    premises = [sentence(p) for p in step.get("premises", [])]
    text = "Because " + "; and ".join(premises) if premises else "Without further premises"
    if step.get("warrant"):
        text += f"; and since {sentence(step['warrant'])}"
    text += f": {sentence(step.get('conclusion'))}"
    if step.get("stated_computation"):
        text += f" ({step['stated_computation']})"
    return text + "."


def render_argument(graph: Graph, order: list[str]) -> str:
    lines = []
    for step_id in order:
        step = graph.steps[step_id]
        lines.append(f"{step_id} [{step.get('move')}]: {render_step(graph, step)}")
    return "\n".join(lines)
