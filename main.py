
from dotenv import load_dotenv
import argparse
import os
import re

import verify as verify_module
import tptp as tptp_module
from clients import make_client, OpenAILLMClient

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_QUESTION = '''\
There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups.
How big is each group of bananas?
'''

# ─── Prompt Templates ─────────────────────────────────────────────────────────

COT_PROMPT = '''\
I will give you a question. Provide a logical argument for the answer. Explain every step of your logical reasoning in the format of a logical syllogism where each premise follows from the previous one, with enough context where I can understand your logic even without the full story. 

End your response with The answer is: X where X is your final answer. X should just be a number, with no units, no formatting, and no trailing explanation. So your formatting 
should be:
EXPLANATION
The answer is: X

Here is the question:

{question}
'''

STRUCTURED_FOL_TEMPLATE = '''\
Convert the following chain-of-thought reasoning into structured first-order logic (FOL).

IMPORTANT: You must faithfully represent the EXACT reasoning steps in the chain of thought below — do not add new inferences, skip steps, or alter the logic. Every inference must correspond directly to a step in the original reasoning.

=== SYNTAX ===
- Quantifiers: ∀ (universal), ∃ (existential)
- Connectives: ∧ (and), ∨ (or), → (implies), ¬ (not)
- Predicates: CamelCase (e.g., InRoom, Holding, FoundDead, MurderLocation)
- Constants: lowercase with underscores (e.g., alice, bob, mr_black, kitchen)
- Variables: single lowercase letters (x, y, z, w, t) — only in quantified formulas

=== OUTPUT CATEGORIES ===
Produce exactly three sections:

## FACTS
Ground atomic formulas taken directly from the question. No quantifiers.
Each fact is one atomic predicate applied to constants.

## RULES
Universally quantified implications that encode external or linguistic knowledge include basic mathematical knowledge.
Every rule MUST start with ∀. These are general principles, not story-specific. 
If they include ground atomic formulas, those formulas must be defined as facts. 

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
- Every inference MUST cite its premises by label.
- Output ONLY the three sections with labeled lines. No extra explanation.

=== EXAMPLE ===
HERE IS EXAMPLE COT REASONING:
Ana has 6 apples. The apples are split into 2 equal bags. Divide 6 by 2 to get 3. So each bag has 3 apples.

CORRECT OUTPUT LOOKS LIKE THIS:
## FACTS
FACT-1: Ana has 6 apples :: TotalApples(n_6)
FACT-2: The apples are split into 2 bags :: AppleBagCount(n_2)

## RULES
RULE-1: If a total number of apples is equally divided into a number of bags with quotient z, then each bag has z apples :: ∀x∀y∀z((TotalApples(x) ∧ AppleBagCount(y) ∧ EqualDivision(x,y,z)) → ApplesPerBag(z))

## INFERENCES
# INF-1: Dividing 6 by 2 gives 3, from FACT-1 and FACT-2 by Arithmetic Computation :: EqualDivision(n_6,n_2,n_3)
# INF-2: Each bag has 3 apples, from FACT-1, FACT-2, INF-1, and RULE-1 by Modus Ponens :: ApplesPerBag(n_3)

=== CHAIN OF THOUGHT REASONING TO CONVERT ===
{cot_text}
'''

COT_REPAIR_TEMPLATE = '''\
The following chain-of-thought reasoning has logical errors identified during verification.
Your job is to produce corrected chain-of-thought reasoning that fixes all the reported issues.

=== ORIGINAL QUESTION ===
{question}

=== ORIGINAL CHAIN OF THOUGHT REASONING ===
{cot_text}

=== ERROR REPORT ===
{error_report}

=== INSTRUCTIONS ===
Revise the chain-of-thought reasoning to fix the errors identified above. Ensure:
- Every fact you state is directly supported by the question.
- Every rule or principle you apply is sound and correctly used.
- Your reasoning steps are logically valid.

Use the same output format as before:
EXPLANATION
The answer is: X
'''

FOL_REPAIR_TEMPLATE = '''\
The following structured first-order logic (FOL) extraction has errors.
Your job is to fix all the reported errors and produce a corrected version. 

IMPORTANT: Your repairs must stay faithful to the EXACT reasoning steps in the original chain of thought — do not add new inferences, skip steps, or alter the logic. Only fix what the error report identifies.

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


# ─── Utilities ────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Extract the last number from model output."""
    num_matches = re.findall(r'[$€£]?\d[\d,]*\.?\d*', text)
    if num_matches:
        return num_matches[-1]
    return ""


def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ─── Plain Prompt (no CoT / FOL) ──────────────────────────────────────────────

def run_plain(question: str, output_dir: str = None, reasoning_client=None) -> dict:
    """
    Ask the model directly — no chain-of-thought, no FOL scaffolding.
    Saves intermediates to output_dir if provided.
    Returns: {answer, response_text}
    """
    if reasoning_client is None:
        reasoning_client = OpenAILLMClient()
    prompt = (
        "Answer the following question. End your answer with 'The answer is: X' "
        f"where X is your final answer.\n\n{question}"
    )
    response_text = reasoning_client.complete(prompt)
    answer = extract_answer(response_text)

    if output_dir:
        _write(os.path.join(output_dir, "plain_prompt.txt"), prompt)
        _write(os.path.join(output_dir, "plain_response.txt"), response_text)
        _write(os.path.join(output_dir, "plain_answer.txt"), answer)

    return {"answer": answer, "response_text": response_text}


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(question: str, max_retries: int = 5, output_dir: str = None,
                 reasoning_client=None, verifier_client=None) -> dict:
    """
    Run the full CoT → Structured FOL → Verify → TPTP pipeline with error-feedback retries.
    Saves all intermediate outputs to output_dir if provided.

    Returns:
        answer          — extracted final answer string
        cot_text        — full chain-of-thought reasoning
        structured_fol  — final (post-repair) structured FOL
        tptp_text       — final TPTP output
        attempts        — number of retry attempts used
        passed_verify   — whether the last FOL passed verification
        passed_tptp     — whether the last TPTP passed prover checks
        verify_reports  — list of verify reports (one per attempt)
        tptp_reports    — list of tptp reports (one per attempt that reached tptp)
    """
    if reasoning_client is None:
        reasoning_client = OpenAILLMClient()
    if verifier_client is None:
        verifier_client = OpenAILLMClient()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _write(os.path.join(output_dir, "question.txt"), question)

    # ── Stage 1: Chain-of-Thought ──────────────────────────────────────────────
    print("=== Stage 1: Chain-of-Thought Reasoning ===")
    cot_text = reasoning_client.complete(COT_PROMPT.format(question=question))
    print(cot_text)

    if output_dir:
        _write(os.path.join(output_dir, "cot.txt"), cot_text)

    # ── Stage 2: Initial FOL Extraction ───────────────────────────────────────
    print("\n=== Stage 2: Initial FOL Extraction ===")
    structured_fol = verifier_client.complete(STRUCTURED_FOL_TEMPLATE.format(cot_text=cot_text))
    print(structured_fol)

    verify_reports = []
    tptp_reports = []
    passed_verify = False
    passed_tptp = False
    tptp_text = ""
    attempts_used = 0

    for attempt in range(1, max_retries + 1):
        attempts_used = attempt
        print(f"\n--- Pipeline Attempt {attempt} of {max_retries} ---")

        if output_dir:
            _write(
                os.path.join(output_dir, f"fol_attempt_{attempt}.txt"),
                structured_fol,
            )

        # ── verify.py ─────────────────────────────────────────────────────────
        print("\n[verify] Checking fact grounding and rule plausibility...")
        verify_errors, verify_report = verify_module.verify_fol(question, structured_fol, client=verifier_client)
        verify_reports.append(verify_report)
        print(verify_report)

        if output_dir:
            _write(
                os.path.join(output_dir, f"verify_attempt_{attempt}.txt"),
                verify_report,
            )

        if verify_errors:
            print(f"\n[verify] Reasoning errors found. Asking reasoning client to produce new CoT...")
            cot_text = reasoning_client.complete(COT_REPAIR_TEMPLATE.format(
                question=question,
                cot_text=cot_text,
                error_report=verify_report,
            ))
            print("\n[repair] Updated CoT:")
            print(cot_text)
            print("\n[repair] Re-extracting FOL from updated CoT...")
            structured_fol = verifier_client.complete(STRUCTURED_FOL_TEMPLATE.format(cot_text=cot_text))
            print(structured_fol)
            if output_dir:
                _write(os.path.join(output_dir, f"cot_repair_attempt_{attempt}.txt"), cot_text)
            continue  # re-verify on next attempt

        passed_verify = True

        # ── tptp.py ───────────────────────────────────────────────────────────
        print("\n[tptp] Converting FOL to TPTP and running prover...")
        tptp_text, tptp_errors, tptp_report = tptp_module.convert_and_check(structured_fol, client=verifier_client)
        tptp_reports.append(tptp_report)
        print("\n[tptp] TPTP output:")
        print(tptp_text)
        print("\n[tptp] Prover report:")
        print(tptp_report)

        if output_dir:
            _write(
                os.path.join(output_dir, f"tptp_attempt_{attempt}.tptp"),
                tptp_text,
            )
            _write(
                os.path.join(output_dir, f"tptp_report_attempt_{attempt}.txt"),
                tptp_report,
            )

        if tptp_errors:
            print(f"\n[tptp] Errors found. Asking LLM to repair...")
            structured_fol = verifier_client.complete(FOL_REPAIR_TEMPLATE.format(
                cot_text=cot_text,
                structured_fol=structured_fol,
                error_report=f"TPTP/Prover errors:\n{tptp_report}",
            ))
            print("\n[repair] Updated FOL:")
            print(structured_fol)
            passed_verify = False  # must re-verify after repair
            continue

        passed_tptp = True
        print("\n=== All checks passed! ===")
        break

    else:
        print(f"\n[warning] Max retries ({max_retries}) reached. Using best available FOL.")

    # ── Save final outputs ────────────────────────────────────────────────────
    if output_dir:
        _write(os.path.join(output_dir, "fol_final.txt"), structured_fol)
        _write(os.path.join(output_dir, "tptp_final.tptp"), tptp_text)
        if verify_reports:
            _write(os.path.join(output_dir, "verify_final.txt"), verify_reports[-1])

    # Also write to SCRIPT_DIR for verify.py / tptp.py standalone compatibility
    with open(os.path.join(SCRIPT_DIR, "story.txt"), "w") as f:
        f.write(question)
    with open(os.path.join(SCRIPT_DIR, "structured_fol.txt"), "w") as f:
        f.write(structured_fol)
    with open(os.path.join(SCRIPT_DIR, "output.tptp"), "w") as f:
        f.write(tptp_text)

    if attempts_used >= max_retries and not (passed_verify and passed_tptp):
        answer = "could not verify the answer"
    else:
        answer = extract_answer(cot_text)
    if output_dir:
        _write(os.path.join(output_dir, "pipeline_answer.txt"), answer)

    print(f"\n=== Final Answer ===")
    print(f"The answer is: {answer}")

    return {
        "answer": answer,
        "cot_text": cot_text,
        "structured_fol": structured_fol,
        "tptp_text": tptp_text,
        "attempts": attempts_used,
        "passed_verify": passed_verify,
        "passed_tptp": passed_tptp,
        "verify_reports": verify_reports,
        "tptp_reports": tptp_reports,
    }


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the CoT → FOL → Verify → TPTP pipeline.",
    )
    parser.add_argument(
        "--question", "-q",
        default=DEFAULT_QUESTION,
        help="The question to answer (defaults to the built-in example).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to write intermediate files (cot.txt, fol_attempt_N.txt, etc.).",
    )
    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=5,
        help="Max verify/tptp repair attempts (default: 5).",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Run plain prompt only (no CoT/FOL pipeline).",
    )
    parser.add_argument(
        "--client", "-c",
        choices=["openai", "kimi", "deepseek"],
        default="openai",
        help="Reasoning client to use for CoT/FOL generation (default: openai). "
             "Verification always uses openai.",
    )
    args = parser.parse_args()

    reasoning_client = make_client(args.client)
    verifier_client = OpenAILLMClient()

    if args.plain:
        result = run_plain(args.question, output_dir=args.output_dir, reasoning_client=reasoning_client)
        print(f"Answer: {result['answer']}")
    else:
        run_pipeline(
            args.question,
            max_retries=args.max_retries,
            output_dir=args.output_dir,
            reasoning_client=reasoning_client,
            verifier_client=verifier_client,
        )
