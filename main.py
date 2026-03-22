
from dotenv import load_dotenv
from openai import OpenAI
import os
import re

import verify as verify_module
import tptp as tptp_module

load_dotenv()
client = OpenAI()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_RETRIES = 3

STORY = '''\
There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups.
How big is each group of bananas?
'''

# ─── Stage 1: Chain-of-Thought Reasoning ──────────────────────────────────────

print("=== Stage 1: Chain-of-Thought Reasoning ===")
cot_response = client.responses.create(
    model="gpt-5-nano",
    input=f'''
I will give you a question. Provide a logical argument for the answer. Explain every step of your logical reasoning in the format of a logical syllogism where each premise follows from the previous one, with enough context where I can understand your logic even without the full story. End your answer with The answer is: X where X is your final answer.

{STORY}
'''
)
cot_text = cot_response.output_text
print(cot_text)

story_path = os.path.join(SCRIPT_DIR, "story.txt")
with open(story_path, "w") as f:
    f.write(STORY)

# ─── FOL Generation Prompt ────────────────────────────────────────────────────

STRUCTURED_FOL_TEMPLATE = '''\
Convert the following chain-of-thought reasoning into structured first-order logic (FOL).

=== SYNTAX ===
- Quantifiers: ∀ (universal), ∃ (existential)
- Connectives: ∧ (and), ∨ (or), → (implies), ¬ (not)
- Predicates: CamelCase (e.g., InRoom, Holding, FoundDead, MurderLocation)
- Constants: lowercase with underscores (e.g., alice, bob, mr_black, kitchen)
- Variables: single lowercase letters (x, y, z, w, t) — only in quantified formulas

=== TIME REPRESENTATION ===
- Use 24-hour time constants: t_21 for 9 PM, t_14 for 2 PM, etc.
- Use named time constants (t_murder, t_seen) ONLY when the exact time is genuinely unspecified.
- Temporal predicates: Before(t1, t2), ShortlyBefore(t1, t2)

=== OUTPUT CATEGORIES ===
Produce exactly three sections:

## FACTS
Ground atomic formulas taken directly from the question. No quantifiers.
Each fact is one atomic predicate applied to constants.

## RULES
Universally quantified implications that encode external or linguistic knowledge.
Every rule MUST start with ∀. These are general principles, not story-specific.

## INFERENCES
Derived conclusions. Each inference MUST cite the FACT/RULE/INF labels it depends on
and name the inference rule used (Modus Ponens, Universal Instantiation, etc.).

=== OUTPUT FORMAT ===
Each line follows this format:
LABEL: <natural language gloss> :: <FOL formula>

Use :: as the delimiter between the gloss and the formula.
Labels are FACT-1, FACT-2, ..., RULE-1, RULE-2, ..., INF-1, INF-2, ...

=== CONSTRAINTS ===
- Do NOT merge a fact and a rule into one formula.
- Do NOT use vague time tokens like "deathtime" — use explicit constants or named constants.
- Every inference MUST cite its premises by label.
- Output ONLY the three sections with labeled lines. No extra explanation.

=== EXAMPLE ===
Given a story: "It was raining at noon. If it rains, the ground is wet."

## FACTS
FACT-1: It was raining at noon :: Raining(t_12)

## RULES
RULE-1: If it rains at a time, the ground is wet at that time :: ∀t (Raining(t) → GroundWet(t))

## INFERENCES
INF-1: From FACT-1, RULE-1 by Modus Ponens: The ground was wet at noon :: GroundWet(t_12)

=== CHAIN OF THOUGHT REASONING TO CONVERT ===
{cot_text}
'''

FOL_REPAIR_TEMPLATE = '''\
The following structured first-order logic (FOL) extraction has errors.
Your job is to fix all the reported errors and produce a corrected version.

=== ORIGINAL CHAIN OF THOUGHT REASONING ===
{cot_text}

=== CURRENT STRUCTURED FOL (with errors) ===
{structured_fol}

=== ERROR REPORT ===
{error_report}

=== INSTRUCTIONS ===
Fix the issues identified in the error report. Ensure:
- Every FACT is directly supported by or obviously entailed from the story.
- Every RULE is a sound commonsense or logical principle.
- Every INFERENCE correctly follows from its cited premises.
- All TPTP syntax issues are corrected.

Use the same output format as before:
LABEL: <natural language gloss> :: <FOL formula>

Output ONLY the three sections (## FACTS, ## RULES, ## INFERENCES) with labeled lines. No extra explanation.
'''


def generate_fol(cot_text: str) -> str:
    prompt = STRUCTURED_FOL_TEMPLATE.format(cot_text=cot_text)
    response = client.responses.create(model="gpt-5-nano", input=prompt)
    return response.output_text


def repair_fol(cot_text: str, structured_fol: str, error_report: str) -> str:
    prompt = FOL_REPAIR_TEMPLATE.format(
        cot_text=cot_text,
        structured_fol=structured_fol,
        error_report=error_report,
    )
    response = client.responses.create(model="gpt-5-nano", input=prompt)
    return response.output_text


def extract_final_answer(text: str) -> str:
    match = re.search(r'[Tt]he answer is[:\s]+(.+)', text)
    if match:
        return match.group(1).strip().rstrip('.')
    return "(answer not found in CoT output)"


# ─── Stage 2+: FOL Extraction with Verify → TPTP feedback loop ────────────────

print("\n=== Stage 2: Initial FOL Extraction ===")
structured_fol = generate_fol(cot_text)
print(structured_fol)

for attempt in range(1, MAX_RETRIES + 1):
    print(f"\n--- Pipeline Attempt {attempt} of {MAX_RETRIES} ---")

    # ── verify.py ──────────────────────────────────────────────────────────────
    print("\n[verify] Checking fact grounding and rule plausibility...")
    verify_errors, verify_report = verify_module.verify_fol(STORY, structured_fol)
    print(verify_report)

    if verify_errors:
        print(f"\n[verify] Errors found on attempt {attempt}. Asking LLM to fix...")
        structured_fol = repair_fol(cot_text, structured_fol, verify_report)
        print("\n[repair] Updated FOL:")
        print(structured_fol)
        # Restart this attempt with the repaired FOL (re-verify before tptp)
        print("\n[verify] Re-checking after repair...")
        verify_errors, verify_report = verify_module.verify_fol(STORY, structured_fol)
        print(verify_report)
        if verify_errors:
            print("[verify] Errors persist after repair; will retry on next attempt.")
            continue

    # ── tptp.py ────────────────────────────────────────────────────────────────
    print("\n[tptp] Converting FOL to TPTP and running prover...")
    tptp_text, tptp_errors, tptp_report = tptp_module.convert_and_check(structured_fol)
    print("\n[tptp] TPTP output:")
    print(tptp_text)
    print("\n[tptp] Prover report:")
    print(tptp_report)

    if tptp_errors:
        print(f"\n[tptp] Errors found on attempt {attempt}. Asking LLM to fix...")
        error_report = f"TPTP/Prover errors:\n{tptp_report}"
        structured_fol = repair_fol(cot_text, structured_fol, error_report)
        print("\n[repair] Updated FOL:")
        print(structured_fol)
        continue

    print("\n=== All checks passed! ===")
    break

else:
    print(f"\n[warning] Max retries ({MAX_RETRIES}) reached. Using best available FOL.")

# ─── Save final outputs ────────────────────────────────────────────────────────

fol_path = os.path.join(SCRIPT_DIR, "structured_fol.txt")
with open(fol_path, "w") as f:
    f.write(structured_fol)
print(f"\nWrote {fol_path}")

# ─── Final Answer ──────────────────────────────────────────────────────────────

print("\n=== Final Answer ===")
answer = extract_final_answer(cot_text)
print(f"The answer is: {answer}")
