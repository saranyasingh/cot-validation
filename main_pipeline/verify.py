
from dotenv import load_dotenv
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

VERIFY_PROMPT = '''\
You are a careful verifier of mathematical reasoning. You are given a math
word problem and a structured extraction containing FACTS (claimed to come
from the problem) and RULES (mathematical principles used in the reasoning).

Your job is to check two things:

**A) Fact Grounding** — Is each FACT directly stated in or obviously entailed
by the problem?
- SUPPORTED: the problem directly states it or it is an obvious direct
  entailment (e.g., "there are 6 apples split into 2 bags" entails
  "the total apple count is 6").
- UNSUPPORTED: the fact adds information not in the problem, changes
  quantities, or makes assumptions beyond what is written.

**B) Rule Validity** — Is each RULE a sound and correctly encoded mathematical
or logical principle?
- PLAUSIBLE: the rule correctly encodes a valid mathematical relationship
  (e.g., division, multiplication, addition) using universally quantified
  predicates with separate variables for each numeric quantity, so that
  the theorem prover can validate specific arithmetic steps as inferences.
- IMPLAUSIBLE: the rule is mathematically incorrect, uses the wrong
  operation, inverts a relationship (e.g., encodes multiplication where
  division is needed), or — critically — embeds a specific arithmetic
  computation directly inside the rule formula rather than abstracting it
  into a predicate relation. For example, a rule that computes "x + y"
  or "x / y" numerically inside the FOL formula is IMPLAUSIBLE: each
  numeric value must be a separate constant and the arithmetic result
  (e.g., Sum(n_5, n_3, n_8), Quotient(n_6, n_2, n_3)) must be left as
  a ground predicate to be asserted as an axiom and checked by the theorem
  prover as an inference, not baked into the rule itself.

=== MATH PROBLEM ===
{story}

=== FACTS ===
{facts}

=== RULES ===
{rules}

For each FACT, respond with exactly one line:
FACT-N: SUPPORTED
or
FACT-N: UNSUPPORTED — <brief explanation>

Then for each RULE, respond with exactly one line:
RULE-N: PLAUSIBLE
or
RULE-N: IMPLAUSIBLE — <brief explanation>

Output ONLY these lines, nothing else.'''


def verify_fol(story_text: str, structured_fol: str, client=None) -> tuple[bool, str]:
    """
    Run fact grounding + rule plausibility checks.
    Returns (has_errors, report_text).
    """
    facts = re.findall(r'^(FACT-\d+):\s*(.+?)\s*::\s*(.+)$', structured_fol, re.MULTILINE)
    rules = re.findall(r'^(RULE-\d+):\s*(.+?)\s*::\s*(.+)$', structured_fol, re.MULTILINE)

    if not facts and not rules:
        return True, "No FACT or RULE lines found in the structured FOL."

    facts_text = "\n".join(f"{label}: {gloss}" for label, gloss, _ in facts)
    rules_text = "\n".join(f"{label}: {gloss}" for label, gloss, _ in rules)

    llm = client if client is not None else _get_default_client()
    report = llm.complete(VERIFY_PROMPT.format(story=story_text, facts=facts_text, rules=rules_text)).strip()

    result_lines = [l.strip() for l in report.splitlines() if l.strip()]
    has_errors = any("UNSUPPORTED" in l or "IMPLAUSIBLE" in l for l in result_lines)

    return has_errors, report


def main():
    story_path = os.path.join(SCRIPT_DIR, "story.txt")
    fol_path = os.path.join(SCRIPT_DIR, "structured_fol.txt")

    for path in (story_path, fol_path):
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run main.py first.")
            raise SystemExit(1)

    with open(story_path) as f:
        story_text = f.read()
    with open(fol_path) as f:
        structured_fol = f.read()

    has_errors, report = verify_fol(story_text, structured_fol)

    print("=== Verification Report ===\n")
    result_lines = [l.strip() for l in report.splitlines() if l.strip()]
    fact_lines = [l for l in result_lines if l.startswith("FACT-")]
    rule_lines = [l for l in result_lines if l.startswith("RULE-")]

    print("--- Fact Grounding (Check A) ---")
    for line in fact_lines:
        print(line)
    unsupported = sum(1 for l in fact_lines if "UNSUPPORTED" in l)
    supported = sum(1 for l in fact_lines if "SUPPORTED" in l and "UNSUPPORTED" not in l)
    print(f"\nFacts: {supported} supported, {unsupported} unsupported")

    print("\n--- Rule Plausibility (Check B) ---")
    for line in rule_lines:
        print(line)
    implausible = sum(1 for l in rule_lines if "IMPLAUSIBLE" in l)
    plausible = sum(1 for l in rule_lines if "PLAUSIBLE" in l and "IMPLAUSIBLE" not in l)
    print(f"\nRules: {plausible} plausible, {implausible} implausible")


if __name__ == "__main__":
    main()
