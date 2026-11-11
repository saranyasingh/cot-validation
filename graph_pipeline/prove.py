"""
Check every step with Vampire. Each step is compiled by `compile_step` into one TFF
problem (premises and warrant as axioms, the conclusion as the conjecture) and proved
on its own. The graph is verified when every step is proved.

E is not used: it rejects the $rat arithmetic that the compiled problems rely on.
"""

import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from graph_pipeline.compile_tff import compile_step
from graph_pipeline.model import Graph
from graph_pipeline.ops import step_order
from main_pipeline.arithmetic import find_vampire

PROVER_TIMEOUT_S = 20

DETAILS = {
    "proved": "the premises and warrant entail the conclusion",
    "not_proved": "the premises and warrant do not entail the conclusion (countermodel found)",
    "contradictory_premises": "the premises and warrant contradict each other, so the step "
                              "proves anything",
    "timeout": "no proof found within the time limit",
}


def run_vampire(vampire: str, problem: str, timeout: int = PROVER_TIMEOUT_S) -> tuple[str, str]:
    """Return (status, detail) for one TPTP problem."""
    with tempfile.NamedTemporaryFile("w", suffix=".p", delete=False) as f:
        f.write(problem)
    try:
        result = subprocess.run(
            [vampire, "--mode", "portfolio", "-t", str(timeout), f.name],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout + 10,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "timeout", DETAILS["timeout"]
    finally:
        os.unlink(f.name)

    statuses = re.findall(r"SZS status (\w+)", output)
    szs = statuses[-1] if statuses else None
    if szs == "Theorem":
        return "proved", DETAILS["proved"]
    if szs in ("CounterSatisfiable", "Satisfiable"):
        return "not_proved", DETAILS["not_proved"]
    if szs in ("ContradictoryAxioms", "Unsatisfiable"):
        return "contradictory_premises", DETAILS["contradictory_premises"]
    error = re.search(r"(?:User error|Parsing error|Parse error)[^\n]*", output)
    if error:
        return "error", error.group(0)
    if szs in (None, "Timeout") and "Time limit" in output:
        return "timeout", DETAILS["timeout"]
    return "unknown", f"prover gave up (SZS status {szs or 'missing'})"


def prove_step(graph: Graph, step_id: str, vampire: str, timeout: int) -> tuple[str, str]:
    try:
        problem = compile_step(graph, step_id)
    except Exception as e:  # a malformed step is reported, not fatal
        return "error", f"could not compile step: {e!r}"
    return run_vampire(vampire, problem, timeout)


def prove_graph(graph: Graph, timeout: int = PROVER_TIMEOUT_S) -> str | None:
    """Set each step's prover and prover_detail. Returns the Vampire path, or None if missing."""
    vampire = find_vampire()
    if vampire is None:
        for step in graph.steps.values():
            step["prover"] = "skipped"
            step["prover_detail"] = "Vampire not found (set VAMPIRE_PATH or put it in tools/)"
        return None
    step_ids = list(graph.steps)
    with ThreadPoolExecutor(max_workers=min(len(step_ids), os.cpu_count() or 1) or 1) as pool:
        results = pool.map(lambda sid: prove_step(graph, sid, vampire, timeout), step_ids)
        for step_id, (status, detail) in zip(step_ids, results):
            graph.steps[step_id]["prover"] = status
            graph.steps[step_id]["prover_detail"] = detail
    return vampire


def prover_verdict(graph: Graph) -> str:
    """VERIFIED if every step is proved, NOT VERIFIED otherwise, SKIPPED if never run."""
    statuses = [s.get("prover") for s in graph.steps.values()]
    if not statuses or any(s in (None, "skipped") for s in statuses):
        return "SKIPPED"
    return "VERIFIED" if all(s == "proved" for s in statuses) else "NOT VERIFIED"


def prover_report(graph: Graph, vampire: str | None) -> str:
    proved = sum(1 for s in graph.steps.values() if s.get("prover") == "proved")
    lines = [f"Prover: {vampire or 'none'}",
             f"Verdict: {prover_verdict(graph)} ({proved}/{len(graph.steps)} steps proved)", ""]
    for step_id in step_order(graph):
        step = graph.steps[step_id]
        lines.append(f"{step_id} ({step.get('move')}): {step.get('prover')}"
                     f" - {step.get('prover_detail')}")
    return "\n".join(lines) + "\n"
