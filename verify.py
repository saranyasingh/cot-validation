
from dotenv import load_dotenv
from openai import OpenAI
import re
import os

load_dotenv()
client = OpenAI()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Read inputs ─────────────────────────────────────────────────────────────

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

# ─── Parse FACT and RULE lines ───────────────────────────────────────────────

facts = re.findall(r'^(FACT-\d+):\s*(.+?)\s*::\s*(.+)$', structured_fol, re.MULTILINE)
rules = re.findall(r'^(RULE-\d+):\s*(.+?)\s*::\s*(.+)$', structured_fol, re.MULTILINE)

if not facts:
    print("Warning: no FACT lines found in structured_fol.txt")
if not rules:
    print("Warning: no RULE lines found in structured_fol.txt")

# ─── Verification: Fact Grounding + Rule Plausibility ────────────────────────

VERIFY_PROMPT = '''\
You are a careful verifier of logical reasoning. You are given a short story
and a structured extraction containing FACTS (claimed to come from the story)
and RULES (general commonsense principles used in the reasoning).

Your job is to check two things:

**A) Fact Grounding** — Is each FACT directly stated in or obviously entailed
by the story?
- SUPPORTED: the story directly states it or it is an obvious direct entailment
  (e.g., "Bob was holding a knife" entails "Bob had a knife").
- UNSUPPORTED: the fact adds information not in the story, changes details,
  or makes assumptions beyond what is written.

**B) Rule Plausibility** — Is each RULE a sound commonsense or logical
principle, given the story and facts it will be applied to?
- PLAUSIBLE: the rule encodes a defensible general principle that does not
  distort the meaning of the facts when applied in this context.
- IMPLAUSIBLE: the rule is logically flawed, overly broad, unfounded,
  a non-sequitur, or — critically — would misrepresent or distort the
  evidence from this story when applied to the facts above.
  For example, a rule that converts incriminating evidence (being at a crime
  scene near the time of the crime) into exonerating evidence (absence from
  the scene) should be marked IMPLAUSIBLE.

=== STORY ===
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

facts_text = "\n".join(f"{label}: {gloss}" for label, gloss, _formula in facts)
rules_text = "\n".join(f"{label}: {gloss}" for label, gloss, _formula in rules)

verify_results = ""
if facts or rules:
    verify_response = client.responses.create(
        model="gpt-5-nano",
        input=VERIFY_PROMPT.format(
            story=story_text, facts=facts_text, rules=rules_text,
        ),
    )
    verify_results = verify_response.output_text.strip()

# ─── Report ──────────────────────────────────────────────────────────────────

print("=== Verification Report ===\n")

if not verify_results:
    print("No facts or rules to check.")
    raise SystemExit(0)

result_lines = [l.strip() for l in verify_results.splitlines() if l.strip()]
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
