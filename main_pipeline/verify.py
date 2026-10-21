
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

Your job is to check two things. Judge MEANING, not formatting: you are looking
for reasoning errors that would make the answer wrong, not for stylistic
problems with how the reasoning was written down. Arithmetic calculations are
checked separately by a theorem prover, so you do not need to recompute them.

**A) Fact Grounding** — Is each FACT directly stated in or obviously entailed
by the problem?
- SUPPORTED: the problem directly states it or it is an obvious direct
  entailment (e.g., "there are 6 apples split into 2 bags" entails
  "the total apple count is 6"). Paraphrases, unit restatements
  ("a dozen" → 12, "half an hour" → 30 minutes), and different but
  equivalent wording all count as SUPPORTED.
- UNSUPPORTED: the fact adds information not in the problem, changes or
  misreads quantities, attaches a quantity to the wrong entity, or makes
  assumptions beyond what is written.

**B) Rule Validity** — Is each RULE a mathematically and logically sound
principle that is correctly applied to this problem?
- PLAUSIBLE: the rule states a valid relationship, however it is written.
- IMPLAUSIBLE: the rule is wrong in substance — it uses the wrong operation,
  inverts a relationship, combines incompatible quantities (e.g. adds GB to
  minutes), double-counts or leaves out a quantity, or misstates how the
  quantities in the problem relate.

Style is NEVER a reason for UNSUPPORTED or IMPLAUSIBLE. Do NOT flag:
- A rule that contains specific numbers instead of variables, if the
  relationship it states is correct.
  e.g. "∀x (Eggs(x, 3) ∧ Days(x, 60) → Total(x, 180))" → PLAUSIBLE
  (3 × 60 really is 180; hard-coded constants are fine).
- A rule that writes arithmetic inline (x + y, n × m) instead of through
  a Sum/Product predicate.
  e.g. "∀a∀b (Cost(a) ∧ Tax(b) → Total(a + b))" → PLAUSIBLE.
- A rule that is narrower than necessary or specific to this story.
  e.g. "If Janet's ducks lay n eggs and she eats k, she has n − k left"
  → PLAUSIBLE.
- Predicate naming, argument order, missing quantifiers on a correct
  formula, or other notation choices.
- A fact that restates a number from the problem in different words.
  e.g. problem says "a dozen eggs", fact says "Eggs(box, 12)" → SUPPORTED.

DO flag (examples of real errors):
- "If x was born n years before y and y is m years old, x is m − n years
  old" → IMPLAUSIBLE (should be m + n).
- "Total daily feed = per-chicken feed ÷ number of chickens"
  → IMPLAUSIBLE (should be ×).
- Problem says the car was in standstill traffic; fact says it drove
  60 mph during that time → UNSUPPORTED.
- Problem says 40% of a 200 GB file was downloaded; fact says 16 GB was
  downloaded → UNSUPPORTED.

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
