
from dotenv import load_dotenv
import subprocess
import tempfile
import shutil
import re
import os

from clients import OpenAILLMClient

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

=== MAPPING ===
- Each FACT-N line becomes: fof(fact_N, axiom, <ground formula>).
- Each RULE-N line becomes: fof(rule_N, axiom, <quantified formula>).
- Each INF-N line becomes:  fof(inf_N, axiom, <derived formula>).
  Preceded by a TPTP comment citing premises: % From fact_N, rule_N by <inference rule>

=== OUTPUT FORMAT ===
- Section headers as TPTP comments: % --- FACTS ---, % --- RULES ---, % --- INFERENCES ---
- One fof(...) statement per line, ending with a period.
- Output ONLY TPTP comments (% ...) and fof() statements. No other text.

=== EXAMPLE ===
Input:
FACT-1: It was raining at noon :: Raining(t_12)
RULE-1: If it rains, ground is wet :: ∀t (Raining(t) → GroundWet(t))
INF-1: From FACT-1, RULE-1 by Modus Ponens: Ground was wet :: GroundWet(t_12)

Output:
% --- FACTS ---
fof(fact_1, axiom, raining(t_12)).

% --- RULES ---
fof(rule_1, axiom, ! [T] : (raining(T) => ground_wet(T))).

% --- INFERENCES ---
% From fact_1, rule_1 by Modus Ponens
fof(inf_1, axiom, ground_wet(t_12)).

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


def convert_and_check(structured_fol: str, client=None) -> tuple[str, bool, str]:
    """
    Convert structured FOL to TPTP and optionally run prover validation.
    Returns (tptp_text, has_errors, error_report).
    """
    llm = client if client is not None else _get_default_client()
    tptp_output = llm.complete(TPTP_TEMPLATE.format(structured_fol=structured_fol))

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
