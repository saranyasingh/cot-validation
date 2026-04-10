"""
rule_pipeline.py — ProofWriter pipeline mirroring main.py.

Mirrors main.py exactly:
  CoT → Structured FOL → Verify → TPTP → repair loop

Two differences from main.py:
  1. The CoT prompt is given the explicit facts (from the problem) and rules
     (from the dataset ruleset) so the model reasons within that closed world.
  2. The verify step checks that FACTS are grounded in the problem statement
     and RULES are grounded in the provided ruleset — not commonsense plausibility.

Generate data first:
    python load_data.py --proofwriter --split test --limit 100
    python rule_pipeline.py --dataset-file proofwriter.json
"""

import argparse
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv

import tptp as tptp_module
from main_pipeline.clients import make_client, OpenAILLMClient

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Prompt Templates ──────────────────────────────────────────────────────────

COT_PROMPT = '''\
You are given a set of facts and a ruleset. Using ONLY these facts and rules,
determine whether the statement is True, False, or Unknown.

Provide a step-by-step logical argument where each step follows from the previous one,
citing which facts and rules you are applying. If the statement cannot be proved or
disproved from the given facts and rules, it is Unknown.

End your answer with: The answer is: True
                  or: The answer is: False
                  or: The answer is: Unknown

=== FACTS (from the problem) ===
{facts}

=== RULES (from the ruleset) ===
{rules}

=== STATEMENT TO EVALUATE ===
{question}
'''

STRUCTURED_FOL_TEMPLATE = '''\
Convert the following chain-of-thought reasoning into structured first-order logic (FOL).

=== SYNTAX ===
- Quantifiers: ∀ (universal), ∃ (existential)
- Connectives: ∧ (and), ∨ (or), → (implies), ¬ (not)
- Predicates: CamelCase (e.g., IsKind, Visits, Needs, Sees)
- Constants: lowercase with underscores (e.g., anne, the_bear, mr_smith)
- Variables: single lowercase letters (x, y, z) — only in quantified formulas

=== OUTPUT CATEGORIES ===
Produce exactly three sections:

## FACTS
Ground atomic formulas taken directly from the problem facts. No quantifiers.
Each fact is one atomic predicate applied to constants.

## RULES
Universally quantified implications taken directly from the given ruleset.
Every rule MUST start with ∀. Do NOT add rules not present in the ruleset.

## INFERENCES
Derived conclusions. Each inference MUST cite the FACT/RULE/INF labels it depends on
and name the inference rule used (Modus Ponens, Universal Instantiation, etc.).
The final inference should establish the answer (True/False/Unknown).

=== OUTPUT FORMAT ===
Each line: LABEL: <natural language gloss> :: <FOL formula>
Labels: FACT-1, FACT-2, ..., RULE-1, RULE-2, ..., INF-1, INF-2, ...

=== CONSTRAINTS ===
- FACTS must come only from the problem facts listed below.
- RULES must come only from the ruleset listed below.
- Do NOT add commonsense or background rules not in the ruleset.
- Every inference MUST cite its premises by label.
- Output ONLY the three sections. No extra explanation.

=== PROBLEM FACTS ===
{facts}

=== RULESET ===
{rules}

=== CHAIN OF THOUGHT REASONING TO CONVERT ===
{cot_text}
'''

FOL_REPAIR_TEMPLATE = '''\
The following structured FOL extraction has errors.
Your job is to fix all reported errors and produce a corrected version.

=== ORIGINAL CHAIN OF THOUGHT REASONING ===
{cot_text}

=== CURRENT STRUCTURED FOL (with errors) ===
{structured_fol}

=== ERROR REPORT ===
{error_report}

=== CONSTRAINTS ===
- FACTS must come only from the problem facts listed below.
- RULES must come only from the ruleset listed below.
- Every INFERENCE must correctly follow from its cited premises.
- All TPTP syntax issues must be corrected.

=== PROBLEM FACTS ===
{facts}

=== RULESET ===
{rules}

Output ONLY the three sections (## FACTS, ## RULES, ## INFERENCES) with labeled lines.
LABEL: <natural language gloss> :: <FOL formula>
No extra explanation.
'''

VERIFY_PROMPT = '''\
You are verifying a structured FOL extraction for a logic puzzle.

**A) Fact Grounding** — Each FACT must come directly from the problem statement.
- SUPPORTED: the fact is directly stated in the problem facts.
- UNSUPPORTED: the fact was invented, modified, or not present in the problem facts.

**B) Rule Grounding** — Each RULE must come directly from the given ruleset.
- GROUNDED: the rule matches a rule in the provided ruleset.
- HALLUCINATED: the rule was invented or is not in the provided ruleset.

=== PROBLEM FACTS (facts must come from here) ===
{problem_facts}

=== RULESET (rules must come from here) ===
{ruleset}

=== EXTRACTED FACTS ===
{facts}

=== EXTRACTED RULES ===
{rules}

For each FACT, respond with exactly one line:
FACT-N: SUPPORTED
or
FACT-N: UNSUPPORTED — <brief explanation>

For each RULE, respond with exactly one line:
RULE-N: GROUNDED
or
RULE-N: HALLUCINATED — <brief explanation>

Output ONLY these lines, nothing else.
'''


# ─── Utilities ─────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Extract True/False/Unknown from 'The answer is: X'."""
    match = re.search(r'[Tt]he answer is[:\s]+(True|False|Unknown)', text)
    if match:
        return match.group(1)
    return "Unknown"


def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ─── Verify (ProofWriter-specific) ─────────────────────────────────────────────

def verify_fol(problem_facts: list[str], ruleset: list[str],
               structured_fol: str, client) -> tuple[bool, str]:
    """
    Check that FACTS are grounded in the problem and RULES are in the ruleset.
    Returns (has_errors, report_text).
    """
    facts  = re.findall(r'^(FACT-\d+):\s*(.+?)\s*::\s*(.+)$', structured_fol, re.MULTILINE)
    rules  = re.findall(r'^(RULE-\d+):\s*(.+?)\s*::\s*(.+)$', structured_fol, re.MULTILINE)

    if not facts and not rules:
        return True, "No FACT or RULE lines found in the structured FOL."

    facts_text = "\n".join(f"{label}: {gloss}" for label, gloss, _ in facts)
    rules_text = "\n".join(f"{label}: {gloss}" for label, gloss, _ in rules)

    report = client.complete(VERIFY_PROMPT.format(
        problem_facts="\n".join(problem_facts),
        ruleset="\n".join(ruleset),
        facts=facts_text,
        rules=rules_text,
    )).strip()

    has_errors = any(
        kw in line
        for line in report.splitlines()
        for kw in ("UNSUPPORTED", "HALLUCINATED")
    )
    return has_errors, report


# ─── Per-Item Pipeline ──────────────────────────────────────────────────────────

def run_item(
    facts: list[str],
    rules: list[str],
    question: str,
    reasoning_client,
    verifier_client,
    max_retries: int = 5,
    output_dir: str = None,
) -> dict:
    """
    CoT → Structured FOL → Verify (fact+rule grounding) → TPTP → repair loop.

    Mirrors main.py's run_pipeline exactly. The only differences:
      - CoT is given the explicit facts and rules.
      - Verify checks grounding against the problem + ruleset, not plausibility.
    """
    facts_str = "\n".join(f"- {f}" for f in facts)
    rules_str = "\n".join(f"- {r}" for r in rules)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _write(os.path.join(output_dir, "question.txt"),  question)
        _write(os.path.join(output_dir, "facts_input.txt"), facts_str)
        _write(os.path.join(output_dir, "rules_input.txt"), rules_str)

    # ── Stage 1: Chain-of-Thought ─────────────────────────────────────────────
    print("=== Stage 1: Chain-of-Thought Reasoning ===")
    cot_text = reasoning_client.complete(COT_PROMPT.format(
        facts=facts_str, rules=rules_str, question=question,
    ))
    print(cot_text)
    if output_dir:
        _write(os.path.join(output_dir, "cot.txt"), cot_text)

    # ── Stage 2: Initial FOL Extraction ──────────────────────────────────────
    print("\n=== Stage 2: Initial FOL Extraction ===")
    structured_fol = reasoning_client.complete(STRUCTURED_FOL_TEMPLATE.format(
        facts=facts_str, rules=rules_str, cot_text=cot_text,
    ))
    print(structured_fol)

    verify_reports = []
    tptp_reports   = []
    passed_verify  = False
    passed_tptp    = False
    tptp_text      = ""
    attempts_used  = 0

    for attempt in range(1, max_retries + 1):
        attempts_used = attempt
        print(f"\n--- Pipeline Attempt {attempt} of {max_retries} ---")

        if output_dir:
            _write(os.path.join(output_dir, f"fol_attempt_{attempt}.txt"), structured_fol)

        # ── Verify: facts from problem, rules from ruleset ────────────────────
        print("\n[verify] Checking fact and rule grounding...")
        verify_errors, verify_report = verify_fol(
            facts, rules, structured_fol, client=verifier_client,
        )
        verify_reports.append(verify_report)
        print(verify_report)
        if output_dir:
            _write(os.path.join(output_dir, f"verify_attempt_{attempt}.txt"), verify_report)

        if verify_errors:
            print("\n[verify] Errors found. Asking LLM to repair...")
            structured_fol = reasoning_client.complete(FOL_REPAIR_TEMPLATE.format(
                cot_text=cot_text,
                structured_fol=structured_fol,
                error_report=verify_report,
                facts=facts_str,
                rules=rules_str,
            ))
            print("\n[repair] Updated FOL:")
            print(structured_fol)
            continue  # re-verify on next attempt

        passed_verify = True

        # ── TPTP: convert and run theorem prover ──────────────────────────────
        print("\n[tptp] Converting FOL to TPTP and running prover...")
        tptp_text, tptp_errors, tptp_report = tptp_module.convert_and_check(
            structured_fol, client=verifier_client,
        )
        tptp_reports.append(tptp_report)
        print("\n[tptp] TPTP output:")
        print(tptp_text)
        print("\n[tptp] Prover report:")
        print(tptp_report)
        if output_dir:
            _write(os.path.join(output_dir, f"tptp_attempt_{attempt}.tptp"), tptp_text)
            _write(os.path.join(output_dir, f"tptp_report_attempt_{attempt}.txt"), tptp_report)

        if tptp_errors:
            print("\n[tptp] Errors found. Asking LLM to repair...")
            structured_fol = reasoning_client.complete(FOL_REPAIR_TEMPLATE.format(
                cot_text=cot_text,
                structured_fol=structured_fol,
                error_report=f"TPTP/Prover errors:\n{tptp_report}",
                facts=facts_str,
                rules=rules_str,
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

    if output_dir:
        _write(os.path.join(output_dir, "fol_final.txt"), structured_fol)
        _write(os.path.join(output_dir, "tptp_final.tptp"), tptp_text)
        if verify_reports:
            _write(os.path.join(output_dir, "verify_final.txt"), verify_reports[-1])

    answer = extract_answer(cot_text)
    if output_dir:
        _write(os.path.join(output_dir, "answer.txt"), answer)

    print(f"\n=== Final Answer: {answer} ===")

    return {
        "answer":         answer,
        "cot_text":       cot_text,
        "structured_fol": structured_fol,
        "tptp_text":      tptp_text,
        "attempts":       attempts_used,
        "passed_verify":  passed_verify,
        "passed_tptp":    passed_tptp,
        "verify_reports": verify_reports,
        "tptp_reports":   tptp_reports,
    }


# ─── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(
    items: list[dict],
    output_dir: str,
    reasoning_client,
    verifier_client,
    max_retries: int = 5,
) -> dict:
    total = len(items)
    correct = passed_verify_count = passed_tptp_count = needed_retry_count = 0
    total_attempts = 0
    results = []

    class_correct = {"True": 0, "False": 0, "Unknown": 0}
    class_total   = {"True": 0, "False": 0, "Unknown": 0}

    print(f"\n{'ID':<22} {'Expected':<10} {'Got':<10} {'Verify':<8} {'TPTP':<6} {'OK'}")
    print("-" * 64)

    for item in items:
        item_id  = item["id"]
        facts    = item["facts"]
        rules    = item["rules"]
        question = item["question"]
        expected = item["expected"]
        item_dir = os.path.join(output_dir, item_id)

        print(f"\n[{item_id}] {question[:70]}")
        print(f"  Expected: {expected} | {len(facts)} facts, {len(rules)} rules")

        result = run_item(
            facts=facts,
            rules=rules,
            question=question,
            reasoning_client=reasoning_client,
            verifier_client=verifier_client,
            max_retries=max_retries,
            output_dir=item_dir,
        )

        predicted = result["answer"]
        ok        = predicted == expected

        if ok:                           correct += 1
        if result["passed_verify"]:      passed_verify_count += 1
        if result["passed_tptp"]:        passed_tptp_count   += 1
        if result["attempts"] > 1:       needed_retry_count  += 1
        total_attempts += result["attempts"]
        class_total[expected]  = class_total.get(expected, 0) + 1
        if ok:
            class_correct[expected] = class_correct.get(expected, 0) + 1

        results.append({
            "id":            item_id,
            "question":      question,
            "expected":      expected,
            "predicted":     predicted,
            "correct":       ok,
            "attempts":      result["attempts"],
            "passed_verify": result["passed_verify"],
            "passed_tptp":   result["passed_tptp"],
        })

        v_m  = "✓" if result["passed_verify"] else "✗"
        t_m  = "✓" if result["passed_tptp"]   else "✗"
        ok_m = "✓" if ok else "✗"
        print(f"\n{item_id:<22} {expected:<10} {predicted:<10} {v_m:<8} {t_m:<6} {ok_m}")

    # ── Summary ───────────────────────────────────────────────────────────────
    acc          = correct / total if total else 0
    verify_rate  = passed_verify_count / total if total else 0
    tptp_rate    = passed_tptp_count   / total if total else 0
    retry_rate   = needed_retry_count  / total if total else 0
    avg_attempts = total_attempts      / total if total else 0

    def pct(n, d):
        return f"{n/d:.1%}" if d else "n/a"

    per_class = {
        cls: {
            "correct":  class_correct.get(cls, 0),
            "total":    class_total.get(cls, 0),
            "accuracy": round(class_correct.get(cls, 0) / class_total[cls], 4)
                        if class_total.get(cls) else 0,
        }
        for cls in ("True", "False", "Unknown")
    }

    summary = {
        "total":           total,
        "correct":         correct,
        "accuracy":        round(acc, 4),
        "passed_verify":   passed_verify_count,
        "verify_rate":     round(verify_rate, 4),
        "passed_tptp":     passed_tptp_count,
        "tptp_rate":       round(tptp_rate, 4),
        "needed_retry":    needed_retry_count,
        "retry_rate":      round(retry_rate, 4),
        "avg_attempts":    round(avg_attempts, 2),
        "per_class":       per_class,
    }

    print("\n" + "=" * 64)
    print(f"{'PROOFWRITER RESULTS':^64}")
    print("=" * 64)
    print(f"  Total items       : {total}")
    print(f"  Correct           : {correct}/{total}  ({acc:.1%})")
    print()
    print(f"  -- Pipeline Internals --")
    print(f"  Passed verify     : {passed_verify_count}/{total}  ({verify_rate:.1%})")
    print(f"  Passed TPTP       : {passed_tptp_count}/{total}  ({tptp_rate:.1%})")
    print(f"  Needed retry      : {needed_retry_count}/{total}  ({retry_rate:.1%})")
    print(f"  Avg attempts/item : {avg_attempts:.2f}")
    print()
    print(f"  -- Per-Class Accuracy --")
    for cls in ("True", "False", "Unknown"):
        n = class_correct.get(cls, 0)
        d = class_total.get(cls, 0)
        print(f"  {cls:<8}: {n}/{d}  ({pct(n, d)})")
    print("=" * 64)

    _write(os.path.join(output_dir, "summary.json"), json.dumps(summary, indent=2))
    _write(os.path.join(output_dir, "results.json"), json.dumps(results, indent=2))
    print(f"\nWrote {os.path.join(output_dir, 'summary.json')}")
    print(f"Wrote {os.path.join(output_dir, 'results.json')}")

    return summary


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ProofWriter pipeline: CoT → FOL → Verify → TPTP (mirrors main.py).",
    )
    parser.add_argument(
        "--dataset-file", "-f", required=True,
        help="ProofWriter JSON from load_data.py --proofwriter.",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Root output directory (default: proofwriter_outputs/).",
    )
    parser.add_argument(
        "--max-retries", "-r", type=int, default=5,
        help="Max verify/TPTP repair attempts per item (default: 5).",
    )
    parser.add_argument(
        "--client", "-c", choices=["openai", "kimi", "deepseek", "vllm"], default="openai",
        help="Reasoning client for CoT/FOL (default: openai). Verification always uses openai.",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Model name for the reasoning client. For vllm, sets VLLM_MODEL (e.g. 'Qwen/Qwen2.5-7B-Instruct').",
    )
    parser.add_argument(
        "--vllm-base-url", default=None,
        help="Base URL for the vLLM server (default: http://localhost:8000/v1). Sets VLLM_BASE_URL.",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Limit number of items to process (default: all).",
    )
    args = parser.parse_args()

    if args.vllm_base_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_base_url
    if args.model:
        if args.client == "vllm":
            os.environ["VLLM_MODEL"] = args.model
        elif args.client == "kimi":
            os.environ["KIMI_MODEL"] = args.model
        elif args.client == "deepseek":
            os.environ["DEEPSEEK_MODEL"] = args.model

    reasoning_client = make_client(args.client)
    verifier_client  = OpenAILLMClient()

    with open(args.dataset_file) as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get("items", raw)
    if args.limit:
        items = items[:args.limit]
    print(f"Loaded {len(items)} items from {args.dataset_file}")

    run_id     = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_dir   = args.output_dir or os.path.join(SCRIPT_DIR, "proofwriter_outputs")
    output_dir = os.path.join(base_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    model_tag = args.model or os.getenv("VLLM_MODEL", "") if args.client == "vllm" else args.model or ""
    print(f"\n=== ProofWriter Benchmark ===")
    print(f"Reasoning client : {args.client}" + (f" ({model_tag})" if model_tag else ""))
    print(f"Verifier client  : openai")
    print(f"Items            : {len(items)}")
    print(f"Max retries      : {args.max_retries}")
    print(f"Output           : {output_dir}\n")

    run_benchmark(
        items=items,
        output_dir=output_dir,
        reasoning_client=reasoning_client,
        verifier_client=verifier_client,
        max_retries=args.max_retries,
    )
