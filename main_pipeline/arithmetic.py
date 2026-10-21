"""
Arithmetic checking for OPERATIONS in the structured FOL.

Each OP line uses one built-in arithmetic predicate whose last argument is the claimed result:
    Sum(a, b, ..., r)        a + b + ... = r
    Difference(a, b, r)      a - b = r
    Product(a, b, ..., r)    a * b * ... = r
    Quotient(a, b, r)        a / b = r

Every operation is translated into a TPTP-TFF problem over $rat and checked with Vampire, which
interprets TPTP arithmetic. Two problems run in parallel per operation: one conjectures the claim,
the other conjectures its negation; whichever is proved decides CORRECT / INCORRECT.

If the claimed result is written with k decimal places, it may differ from the exact value by up
to half a unit in the last place (so 40/3 = 13.33 is accepted). Integer results must be exact.

If Vampire is not available, the same check is done with exact Python fractions.
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

OPERATORS = {
    "sum":        ("$sum",        "+", None),
    "difference": ("$difference", "-", 3),
    "product":    ("$product",    "*", None),
    "quotient":   ("$quotient",   "/", 3),
}

OP_LINE_RE = re.compile(r'^(OP-\d+):\s*(.+?)\s*::\s*(.+)$')
NUMBER_RE = re.compile(r'^-?\d+(?:\.\d+)?(?:/\d+)?$')
PROVER_TIMEOUT_S = 5


def find_vampire():
    for path in (os.getenv("VAMPIRE_PATH"), shutil.which("vampire"),
                 os.path.join(REPO_ROOT, "tools", "vampire")):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def parse_operations(structured_fol: str) -> list[tuple[str, str, str]]:
    """Return [(label, gloss, formula), ...] for every OP-N line."""
    ops = []
    for raw_line in structured_fol.splitlines():
        m = OP_LINE_RE.match(raw_line.strip().lstrip('#').strip())
        if m:
            ops.append((m.group(1), m.group(2), m.group(3)))
    return ops


def _parse_number(token: str):
    token = token.strip().replace(",", "")
    token = re.sub(r'^n_(neg_)?', lambda m: "-" if m.group(1) else "", token)
    if re.match(r'^-?\d+_\d+$', token):
        token = token.replace("_", ".")
    if not NUMBER_RE.match(token):
        return None, 0
    decimals = len(token.split(".")[1].split("/")[0]) if "." in token else 0
    return Fraction(token), decimals


def parse_operation(formula: str):
    """Parse 'Product(3, 68, 204)' → ('product', [3, 68], 204, decimals_of_result).

    Raises ValueError with a readable message if the formula is not a well-formed operation.
    """
    m = re.match(r'^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$', formula)
    if not m:
        raise ValueError(f"not of the form Operator(args): {formula}")
    op = m.group(1).lower()
    if op not in OPERATORS:
        raise ValueError(f"unknown operator '{m.group(1)}' (use Sum, Difference, Product, Quotient)")
    args = [a for a in m.group(2).split(",") if a.strip()]
    arity = OPERATORS[op][2]
    if len(args) < 3 or (arity and len(args) != arity):
        need = f"exactly {arity}" if arity else "at least 3"
        raise ValueError(f"{m.group(1)} needs {need} arguments, got {len(args)}")
    values = []
    decimals = 0
    for a in args:
        value, decimals = _parse_number(a)
        if value is None:
            raise ValueError(f"argument '{a.strip()}' is not a number")
        values.append(value)
    return op, values[:-1], values[-1], decimals


def _evaluate(op: str, operands: list[Fraction]) -> Fraction:
    result = operands[0]
    for x in operands[1:]:
        if op == "sum":
            result += x
        elif op == "difference":
            result -= x
        elif op == "product":
            result *= x
        else:
            result /= x
    return result


def _tolerance(decimals: int) -> Fraction:
    return Fraction(1, 2 * 10 ** decimals) if decimals else Fraction(0)


def _rat(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def to_tptp(op: str, operands: list[Fraction], claimed: Fraction, decimals: int) -> str:
    """TFF formula asserting the operation's claim, over $rat."""
    fn = OPERATORS[op][0]
    lhs = _rat(operands[0])
    for x in operands[1:]:
        lhs = f"{fn}({lhs}, {_rat(x)})"
    rhs = _rat(claimed)
    tol = _tolerance(decimals)
    if tol == 0:
        return f"{lhs} = {rhs}"
    return (f"($lesseq($difference({lhs}, {rhs}), {_rat(tol)}) & "
            f"$lesseq($difference({rhs}, {lhs}), {_rat(tol)}))")


def _run_vampire(vampire: str, claim: str) -> str:
    """Return 'CORRECT', 'INCORRECT', or 'UNKNOWN' for a TFF claim."""
    procs = {}
    outputs = {}
    paths = []
    try:
        for verdict, formula in (("CORRECT", claim), ("INCORRECT", f"~({claim})")):
            with tempfile.NamedTemporaryFile("w", suffix=".p", delete=False) as f:
                f.write(f"tff(op, conjecture, {formula}).\n")
            out = tempfile.NamedTemporaryFile("w+", suffix=".out", delete=False)
            paths += [f.name, out.name]
            outputs[verdict] = out
            procs[verdict] = subprocess.Popen(
                [vampire, "-t", str(PROVER_TIMEOUT_S), f.name],
                stdout=out, stderr=subprocess.STDOUT,
            )
        deadline = time.monotonic() + PROVER_TIMEOUT_S + 2
        pending = dict(procs)
        while pending and time.monotonic() < deadline:
            for verdict, p in list(pending.items()):
                if p.poll() is None:
                    continue
                del pending[verdict]
                outputs[verdict].seek(0)
                if re.search(r'SZS status Theorem', outputs[verdict].read()):
                    return verdict
            time.sleep(0.01)
        return "UNKNOWN"
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
            p.wait()
        for out in outputs.values():
            out.close()
        for path in paths:
            os.unlink(path)


def _fmt(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{float(x):.6g}"


def check_operations(structured_fol: str) -> tuple[bool, str, str]:
    """
    Check every OP-N line. Returns (has_errors, report, tptp_text).

    Report lines look like:
        OP-1: CORRECT
        OP-2: INCORRECT — 204 + 160 + 330 = 694, not 794
        OP-3: MALFORMED — argument 'x' is not a number
    """
    ops = parse_operations(structured_fol)
    if not ops:
        return False, "No OPERATIONS found.", ""

    vampire = find_vampire()
    lines = []
    tptp_lines = []
    has_errors = False

    for label, _gloss, formula in ops:
        try:
            op, operands, claimed, decimals = parse_operation(formula)
        except ValueError as e:
            lines.append(f"{label}: MALFORMED — {e}")
            has_errors = True
            continue

        symbol = OPERATORS[op][1]
        expr = f" {symbol} ".join(_fmt(x) for x in operands)

        if op == "quotient" and operands[1] == 0:
            lines.append(f"{label}: INCORRECT — division by zero in {expr}")
            has_errors = True
            continue

        claim = to_tptp(op, operands, claimed, decimals)
        tptp_lines.append(f"% {label}: {formula}")
        tptp_lines.append(f"tff({label.lower().replace('-', '_')}, conjecture, {claim}).")

        if vampire:
            status = _run_vampire(vampire, claim)
        else:
            exact = _evaluate(op, operands)
            status = "CORRECT" if abs(exact - claimed) <= _tolerance(decimals) else "INCORRECT"

        if status == "CORRECT":
            lines.append(f"{label}: CORRECT")
        elif status == "INCORRECT":
            exact = _evaluate(op, operands)
            lines.append(f"{label}: INCORRECT — {expr} = {_fmt(exact)}, not {_fmt(claimed)}")
            has_errors = True
        else:
            lines.append(f"{label}: UNKNOWN — prover could not decide {expr} = {_fmt(claimed)}")
            has_errors = True

    checker = "vampire" if vampire else "python (vampire not found)"
    report = f"Arithmetic checked with {checker}\n" + "\n".join(lines)
    return has_errors, report, "\n".join(tptp_lines)
