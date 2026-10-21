
from dotenv import load_dotenv
import subprocess
import tempfile
import shutil
import re
import os

from main_pipeline.clients import OpenAILLMClient

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_default_client = None


def _get_default_client():
    global _default_client
    if _default_client is None:
        _default_client = OpenAILLMClient()
    return _default_client

TPTP_TEMPLATE = '''\
Convert the following structured first-order logic into TPTP-FOF syntax.
IMPORTANT: You must faithfully represent the EXACT reasoning steps in the chain of thought below — do not add new inferences, skip steps, or alter the logic. 
The goal is to produce a TPTP file that exactly captures the entire reasoning process as given.


=== INPUT (Structured FOL) ===
{structured_fol}

=== TPTP-FOF SYNTAX RULES ===
- Universal quantifier: ! [X] : (...)
- Existential quantifier: ? [X] : (...)
- And: &
- Or: |
- Implies: =>
- Not: ~
- Variables: UPPERCASE (X, Y, Z, W, T)
- Constants: lowercase with underscores (alice, bob, mr_black, kitchen)
- Predicates: lowercase snake_case (in_room, holding, found_dead, murder_location)
- Time constants stay as-is: t_21, t_murder, t_seen, etc.

=== NUMBERS (IMPORTANT) ===
NEVER write a bare number such as 4, 204, or 1.2 anywhere in a formula. Provers read bare numbers as typed
integers, which clash with the untyped variables in the rules ("Type mismatch ... expected $int but got $i")
and stop the prover from running at all. Write every number as a constant with the prefix n_:
- integers: 4 → n_4, 204 → n_204
- decimals: replace the point with an underscore: 1.2 → n_1_2, 0.25 → n_0_25
- negatives: -5 → n_neg_5
- drop units, $, %, and thousands separators first: $1,500 → n_1500
Use exactly the same constant for the same number everywhere in the file.

=== OTHER COMMON ERRORS TO AVOID ===
- Use only fof(...). Do NOT use tff, thf, or arithmetic symbols ($sum, $product, $int, <, >, +, -, *, /),
  and do not invent arithmetic functions such as minus(N, M) or multiply(N, M).
  Arithmetic is expressed only through the predicates sum, difference, product, quotient.
- If a rule computes a value inline, rewrite the computation as an extra condition using those predicates, so
  that the matching op_N axiom can satisfy it:
  "∀x (Downloads(m1, x) → Downloads(m2, 3*x))"
    → fof(rule_1, axiom, ! [X,Y] : ((downloads(m1, X) & product(n_3, X, Y)) => downloads(m2, Y))).
- Every variable must be bound by a quantifier; every quantifier lists its variables in brackets: ! [X,Y] : (...)
- Each predicate must be used with the same number of arguments everywhere.
- Constants and predicates must start with a lowercase letter and contain only letters, digits, and underscores.
- Every statement ends with ")." and parentheses must balance.

=== MAPPING ===
- Each FACT-N line becomes: fof(fact_N, axiom, <ground formula>).
- Each OP-N line becomes:   fof(op_N, axiom, <ground formula>). (e.g. Product(3, 68, 204) → product(n_3, n_68, n_204))
- Each RULE-N line becomes: fof(rule_N, axiom, <quantified formula>).
- Each INF-N line becomes:  fof(inf_N, axiom, <derived formula>).
  Preceded by a TPTP comment citing premises: % From fact_N, op_N, rule_N by <inference rule>

=== OUTPUT FORMAT ===
- Section headers as TPTP comments: % --- FACTS ---, % --- OPERATIONS ---, % --- RULES ---, % --- INFERENCES ---
- One fof(...) statement per line, ending with a period.
- Output ONLY TPTP comments (% ...) and fof() statements. No other text.

=== EXAMPLE ===
Input:
FACT-1: Sam buys 3 boxes of pencils :: Boxes(sam, 3)
FACT-2: Each box holds 12 pencils :: PerBox(sam, 12)
OP-1: 3 times 12 is 36 :: Product(3, 12, 36)
RULE-1: If x bought n boxes of m pencils and n × m = p, x has p pencils :: ∀x∀n∀m∀p (Boxes(x,n) ∧ PerBox(x,m) ∧ Product(n,m,p) → Has(x,p))
INF-1: Sam has 36 pencils, from FACT-1, FACT-2, OP-1, and RULE-1 by Modus Ponens :: Has(sam, 36)

Output:
% --- FACTS ---
fof(fact_1, axiom, boxes(sam, n_3)).
fof(fact_2, axiom, per_box(sam, n_12)).

% --- OPERATIONS ---
fof(op_1, axiom, product(n_3, n_12, n_36)).

% --- RULES ---
fof(rule_1, axiom, ! [X,N,M,P] : ((boxes(X, N) & per_box(X, M) & product(N, M, P)) => has(X, P))).

% --- INFERENCES ---
% From fact_1, fact_2, op_1, rule_1 by Modus Ponens
fof(inf_1, axiom, has(sam, n_36)).

=== CONVERT NOW ===
'''


def find_prover():
    """Find an available TPTP-compatible theorem prover."""
    provers = [
        ("eprover", ["eprover", "--auto", "--tptp3-format", "-s"]),
        ("vampire", ["vampire", "--input_syntax", "tptp"]),
    ]
    for name, args in provers:
        if shutil.which(args[0]):
            return name, args
    return None, None


def parse_tptp(tptp_text):
    """Split TPTP text into premise fof() lines and (comment, fof) inference pairs."""
    axioms = []
    inferences = []

    lines = tptp_text.strip().split('\n')
    pending_comment = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('%'):
            if re.search(r'[Ff]rom\s+', stripped):
                pending_comment = stripped
            continue
        if stripped.startswith('fof('):
            if re.match(r'fof\(inf_\d+', stripped):
                inferences.append((pending_comment, stripped))
            else:
                axioms.append(stripped)
            pending_comment = None

    return axioms, inferences


def _number_constant(m: re.Match) -> str:
    token = m.group(0)
    sign = "neg_" if token.startswith("-") else ""
    return f"n_{sign}{token.lstrip('-').replace('.', '_')}"


def normalize_numbers(tptp_text: str) -> str:
    """Rewrite any bare numbers left in fof() lines as n_ constants (see NUMBERS in the prompt)."""
    out = []
    for line in tptp_text.splitlines():
        if line.strip().startswith("fof("):
            line = re.sub(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])', _number_constant, line)
        out.append(line)
    return "\n".join(out)


def convert_and_check(structured_fol: str, client=None) -> tuple[str, bool, str]:
    """
    Convert structured FOL to TPTP and optionally run prover validation.
    Returns (tptp_text, has_errors, error_report).
    """
    llm = client if client is not None else _get_default_client()
    tptp_output = normalize_numbers(llm.complete(TPTP_TEMPLATE.format(structured_fol=structured_fol)))

    tptp_path = os.path.join(SCRIPT_DIR, "output.tptp")
    with open(tptp_path, "w") as f:
        f.write(tptp_output)

    name, base_args = find_prover()
    if name is None:
        return tptp_output, False, "No theorem prover available; skipping formal validation."

    # Syntax check (eprover only)
    if name == "eprover":
        try:
            syntax_check = subprocess.run(
                ["eprover", "--syntax-only", "--tptp3-format", tptp_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            )
            if syntax_check.returncode != 0:
                error_msg = (syntax_check.stderr or syntax_check.stdout).strip()
                return tptp_output, True, f"TPTP syntax error:\n{error_msg}"
        except subprocess.TimeoutExpired:
            return tptp_output, True, "TPTP syntax check timed out (possible malformed formula or unsupported function)"

    axioms, inferences = parse_tptp(tptp_output)

    if not inferences:
        return tptp_output, False, "No inferences found to validate."

    errors = []
    inference_results = []
    proven_so_far = []

    for comment, fof_line in inferences:
        match = re.match(r'fof\((inf_\d+),\s*\w+,\s*(.+)\)\.\s*$', fof_line)
        if not match:
            errors.append(f"Could not parse inference line: {fof_line}")
            continue

        inf_name = match.group(1)
        formula = match.group(2)

        problem_lines = axioms + proven_so_far + [f"fof({inf_name}, conjecture, {formula})."]
        problem_text = '\n'.join(problem_lines)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.tptp', delete=False, dir=SCRIPT_DIR
        ) as tmp:
            tmp.write(problem_text)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                base_args + [tmp_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            output = result.stdout + result.stderr

            if re.search(r'SZS status\s+Theorem', output):
                status = "PROVED"
            elif re.search(r'SZS status\s+(CounterSatisfiable|Satisfiable)', output):
                status = "NOT PROVED"
                errors.append(f"{inf_name}: NOT PROVED — inference does not follow from axioms")
            elif re.search(r'SZS status\s+Timeout', output):
                status = "TIMEOUT"
            elif re.search(r'SZS status\s+', output):
                szs = re.search(r'SZS status\s+(\w+)', output).group(1)
                status = f"UNKNOWN ({szs})"
            else:
                status = "UNKNOWN"

            prefix = f"  {comment}\n" if comment else ""
            inference_results.append(f"{prefix}  {inf_name}: {status}")

        except subprocess.TimeoutExpired:
            inference_results.append(f"  {inf_name}: TIMEOUT")
        except Exception as e:
            err = f"{inf_name}: ERROR ({e})"
            inference_results.append(f"  {err}")
            errors.append(err)
        finally:
            os.unlink(tmp_path)

        proven_so_far.append(fof_line)

    report = "\n".join(inference_results)
    has_errors = bool(errors)
    if has_errors:
        report += "\n\nProver errors:\n" + "\n".join(errors)

    return tptp_output, has_errors, report


def main():
    fol_path = os.path.join(SCRIPT_DIR, "structured_fol.txt")
    if not os.path.exists(fol_path):
        print(f"Error: {fol_path} not found. Run main.py first.")
        raise SystemExit(1)

    with open(fol_path) as f:
        structured_fol = f.read()

    print("=== Structured FOL (from file) ===")
    print(structured_fol)

    tptp_output, has_errors, report = convert_and_check(structured_fol)

    print("\n=== First Order Logic (TPTP) ===")
    print(tptp_output)

    tptp_path = os.path.join(SCRIPT_DIR, "output.tptp")
    print(f"Wrote {tptp_path}")

    print("\n=== Theorem Prover Validation ===")
    print(report)


if __name__ == "__main__":
    main()
