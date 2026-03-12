
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
import tempfile
import shutil
import re
import os

load_dotenv()
client = OpenAI()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Read structured FOL from main.py output ─────────────────────────────────

fol_path = os.path.join(SCRIPT_DIR, "structured_fol.txt")
if not os.path.exists(fol_path):
    print(f"Error: {fol_path} not found. Run main.py first.")
    raise SystemExit(1)

with open(fol_path) as f:
    structured_fol = f.read()

print("=== Structured FOL (from file) ===")
print(structured_fol)

# ─── Stage 3: Convert Structured FOL → TPTP-FOF ─────────────────────────────

TPTP_TEMPLATE = '''\
Convert the following structured first-order logic into TPTP-FOF syntax.

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

tptp_prompt = TPTP_TEMPLATE.format(structured_fol=structured_fol)
tptp_response = client.responses.create(model="gpt-5-nano", input=tptp_prompt)

tptp_output = tptp_response.output_text
print("\n=== First Order Logic (TPTP) ===")
print(tptp_output)

tptp_path = os.path.join(SCRIPT_DIR, "output.tptp")
with open(tptp_path, "w") as f:
    f.write(tptp_output)
print(f"Wrote {tptp_path}")

# ─── Stage 4: Theorem Prover Validation ──────────────────────────────────────


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


def run_prover(tptp_file):
    """Validate each inference by proving it as a conjecture from the axioms."""
    name, base_args = find_prover()
    if name is None:
        print("\n=== Theorem Prover ===")
        print("No TPTP theorem prover found. Install one with:")
        print("  brew install eprover    # E Theorem Prover")
        print("  # or download Vampire from https://vprover.github.io/")
        return

    print(f"\n=== Theorem Prover Validation ({name}) ===")

    with open(tptp_file) as f:
        tptp_text = f.read()

    # First, syntax-check the entire file
    if name == "eprover":
        syntax_check = subprocess.run(
            ["eprover", "--syntax-only", "--tptp3-format", tptp_file],
            capture_output=True, text=True, timeout=10,
        )
        if syntax_check.returncode != 0:
            print("TPTP syntax error:")
            print(syntax_check.stderr or syntax_check.stdout)
            return
        print("Syntax check passed.")

    axioms, inferences = parse_tptp(tptp_text)

    if not inferences:
        print("No inferences found to validate.")
        return

    print(f"Found {len(axioms)} axiom(s), {len(inferences)} inference(s) to validate.\n")

    proven_so_far = []
    for comment, fof_line in inferences:
        # Extract inference name and formula from: fof(inf_N, axiom, <formula>).
        match = re.match(r'fof\((inf_\d+),\s*\w+,\s*(.+)\)\.\s*$', fof_line)
        if not match:
            print(f"  Could not parse: {fof_line}")
            continue

        inf_name = match.group(1)
        formula = match.group(2)

        # Build problem: all axioms + previously proven inferences + this as conjecture
        problem_lines = axioms + proven_so_far
        problem_lines.append(f"fof({inf_name}, conjecture, {formula}).")
        problem_text = '\n'.join(problem_lines)

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.tptp', delete=False, dir=SCRIPT_DIR
        ) as tmp:
            tmp.write(problem_text)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                base_args + [tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout + result.stderr

            if re.search(r'SZS status\s+Theorem', output):
                status = "PROVED"
            elif re.search(r'SZS status\s+(CounterSatisfiable|Satisfiable)', output):
                status = "NOT PROVED"
            elif re.search(r'SZS status\s+Timeout', output):
                status = "TIMEOUT"
            elif re.search(r'SZS status\s+', output):
                szs = re.search(r'SZS status\s+(\w+)', output).group(1)
                status = f"UNKNOWN ({szs})"
            else:
                status = "UNKNOWN"

            prefix = f"  {comment}\n" if comment else ""
            print(f"{prefix}  {inf_name}: {status}")

        except subprocess.TimeoutExpired:
            print(f"  {inf_name}: TIMEOUT (30s)")
        except Exception as e:
            print(f"  {inf_name}: ERROR ({e})")
        finally:
            os.unlink(tmp_path)

        # Add as axiom for subsequent proofs regardless of status
        proven_so_far.append(fof_line)


run_prover(tptp_path)
