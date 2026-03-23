"""
rule_pipeline.py — Ruleset-constrained proof pipeline on ProofWriter.

Key difference from main.py:
  - RULES are extracted only from the given theory — no LLM-invented rules.
  - Verification checks that every extracted rule is explicitly grounded in the
    theory text before the proof is generated.
  - The LLM then proves/disproves statements using ONLY those grounded rules.
  - Benchmarks True/False/Unknown classification accuracy on ProofWriter.

Usage:
    python rule_pipeline.py
    python rule_pipeline.py --client kimi --depth 1 --limit 20
    python rule_pipeline.py --dataset-file my_data.json
    python rule_pipeline.py --output-dir ./pw_run
"""

import argparse
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset

from clients import make_client, OpenAILLMClient

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

HF_DATASET = "D3xter1922/proofwriter-dataset"

# ── Prompt Templates ────────────────────────────────────────────────────────────

THEORY_TO_FOL_PROMPT = '''\
You are given a logical theory. Extract its facts and rules into structured first-order logic.

=== CRITICAL CONSTRAINT ===
- FACTS must come ONLY from explicit statements in the theory (not rules/conditionals).
- RULES must come ONLY from conditional statements explicitly present in the theory.
- Do NOT add commonsense knowledge, background rules, or anything not written in the theory.
- Every RULE must be directly traceable to a sentence in the theory.

=== SYNTAX ===
- Quantifiers: ∀ (universal), ∃ (existential)
- Connectives: ∧ (and), ∨ (or), → (implies), ¬ (not)
- Predicates: CamelCase (e.g., IsYoung, IsBig, IsKind, Chases)
- Constants: lowercase with underscores (e.g., bald_eagle, anne, the_cat)
- Variables: single lowercase letters (x, y, z)

=== OUTPUT CATEGORIES ===

## FACTS
Ground atomic formulas directly from the theory. No quantifiers. One per line.

## RULES
Universally quantified implications from the theory's conditional statements.
Every rule MUST start with ∀. One per line.

=== OUTPUT FORMAT ===
LABEL: <exact natural language from theory> :: <FOL formula>

Labels: FACT-1, FACT-2, ..., RULE-1, RULE-2, ...
Output ONLY the two sections. No extra explanation.

=== EXAMPLE ===
Theory: "Anne is young. The cat is big. If something is young then it is kind."

## FACTS
FACT-1: Anne is young :: IsYoung(anne)
FACT-2: The cat is big :: IsBig(the_cat)

## RULES
RULE-1: If something is young then it is kind :: ∀x (IsYoung(x) → IsKind(x))

=== THEORY ===
{theory}
'''

RULE_GROUNDING_PROMPT = '''\
Check whether each extracted RULE is explicitly stated in the theory as a conditional.
Mark rules that were invented or generalised beyond what the theory says as HALLUCINATED.

=== THEORY ===
{theory}

=== EXTRACTED RULES ===
{rules}

For each rule respond with exactly one line:
RULE-N: GROUNDED — <the sentence in the theory that supports this rule>
or
RULE-N: HALLUCINATED — <brief explanation of what was added that isn't in the theory>

Output ONLY these lines.
'''

RULE_REPAIR_PROMPT = '''\
The following rule extraction from a theory contains hallucinated rules not present in the theory.
Fix the extraction so that every RULE is directly grounded in the theory.

=== THEORY ===
{theory}

=== CURRENT EXTRACTION (with errors) ===
{facts_and_rules}

=== GROUNDING ERRORS ===
{error_report}

Remove or correct all HALLUCINATED rules. Keep all FACTS as-is.
Output ONLY the corrected ## FACTS and ## RULES sections in the same format.
'''

CONSTRAINED_PROOF_PROMPT = '''\
You are given a verified set of FACTS and RULES extracted from a theory, and a statement to evaluate.

=== CRITICAL CONSTRAINT ===
Use ONLY the FACTS and RULES listed below — no new rules, no background knowledge.
If the statement cannot be proved or disproved from these alone, answer Unknown.

=== FACTS AND RULES ===
{facts_and_rules}

=== STATEMENT TO EVALUATE ===
{question}

=== INSTRUCTIONS ===
1. Convert the statement to FOL.
2. Attempt to prove it step by step, citing only the FACT/RULE labels above.
3. If you cannot prove it, attempt to prove its negation.
4. Conclude True (proved), False (negation proved), or Unknown (neither proved).

=== OUTPUT FORMAT ===
## INFERENCES
INF-1: <gloss> :: <FOL formula>  [From FACT-X, RULE-Y by <inference rule>]
INF-2: ...

ANSWER: True
(or False, or Unknown)

Output ONLY the ## INFERENCES section and the ANSWER line.
'''


# ── Data Loading ────────────────────────────────────────────────────────────────

def _parse_row(en: str, ro: str, idx: int) -> dict | None:
    """
    Parse one row from D3xter1922/proofwriter-dataset.

    en field: "$answer$ ; $proof$ ; $question$ = <q> ; $context$ = sent1: ... sentN: ..."
    ro field: "$answer$ = True/False/Unknown ; $proof$ = sentX"
    """
    # Extract question from en
    q_match = re.search(r'\$question\$\s*=\s*(.+?)\s*;', en)
    if not q_match:
        return None
    question = q_match.group(1).strip().rstrip('.')

    # Extract context (theory) from en — everything after "$context$ ="
    ctx_match = re.search(r'\$context\$\s*=\s*(.+)$', en, re.DOTALL)
    if not ctx_match:
        return None
    # Sentences are labelled "sentN: <text>." — join them as plain prose
    raw_ctx = ctx_match.group(1).strip()
    sentences = re.findall(r'sent\d+:\s*(.+?)(?=\s+sent\d+:|$)', raw_ctx)
    theory = " ".join(s.strip().rstrip('.') + '.' for s in sentences) if sentences else raw_ctx

    # Extract answer from ro
    ans_match = re.search(r'\$answer\$\s*=\s*(True|False|Unknown)', ro, re.IGNORECASE)
    expected = ans_match.group(1).capitalize() if ans_match else "Unknown"

    return {
        "id": f"item_{idx:05d}",
        "theory": theory,
        "question": question,
        "expected": expected,
    }


def fetch_proofwriter(split: str = "test", limit: int = 50) -> list[dict]:
    """Load ProofWriter items using the datasets library."""
    print(f"Loading {HF_DATASET} ({split} split, up to {limit} items)...")
    ds = load_dataset(HF_DATASET, split=split)

    items = []
    for i, row in enumerate(ds):
        if len(items) >= limit:
            break
        translation = row.get("translation", {})
        en = translation.get("en", "")
        ro = translation.get("ro", "")
        parsed = _parse_row(en, ro, i + 1)
        if parsed:
            items.append(parsed)

    print(f"Loaded {len(items)} items.")
    return items


# ── Utilities ────────────────────────────────────────────────────────────────────

def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def extract_answer(text: str) -> str:
    """Parse ANSWER: True/False/Unknown from proof output."""
    match = re.search(r'^ANSWER:\s*(True|False|Unknown)', text, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "Unknown"


def extract_rules(facts_and_rules: str) -> str:
    """Return only the RULE-N lines from a facts+rules block."""
    lines = [l.strip() for l in facts_and_rules.splitlines() if re.match(r'RULE-\d+:', l.strip())]
    return "\n".join(lines) if lines else "(no rules)"


# ── Per-Item Pipeline ────────────────────────────────────────────────────────────

def run_item(
    theory: str,
    question: str,
    reasoning_client,
    verifier_client,
    max_retries: int = 3,
    output_dir: str = None,
) -> dict:
    """
    Constrained proof pipeline for one ProofWriter item.

    Stage 1: Extract facts + rules from theory (reasoning client)
    Stage 2: Verify all rules are grounded in theory (verifier client); repair if not
    Stage 3: Generate proof using only verified rules (reasoning client)

    Returns: {answer, facts_and_rules, proof, rule_attempts, passed_grounding}
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _write(os.path.join(output_dir, "theory.txt"), theory)
        _write(os.path.join(output_dir, "question.txt"), question)

    # ── Stage 1: Theory → Structured FOL ──────────────────────────────────────
    print("  [stage 1] Extracting facts and rules from theory...")
    facts_and_rules = reasoning_client.complete(
        THEORY_TO_FOL_PROMPT.format(theory=theory)
    )
    print(facts_and_rules)

    if output_dir:
        _write(os.path.join(output_dir, "facts_and_rules_raw.txt"), facts_and_rules)

    # ── Stage 2: Verify rule grounding with repair loop ────────────────────────
    grounding_reports = []
    passed_grounding = False
    rule_attempts = 0

    for attempt in range(1, max_retries + 1):
        rule_attempts = attempt
        rules_text = extract_rules(facts_and_rules)

        print(f"  [stage 2, attempt {attempt}] Checking rule grounding...")
        grounding_report = verifier_client.complete(
            RULE_GROUNDING_PROMPT.format(theory=theory, rules=rules_text)
        ).strip()
        grounding_reports.append(grounding_report)
        print(grounding_report)

        if output_dir:
            _write(
                os.path.join(output_dir, f"grounding_attempt_{attempt}.txt"),
                grounding_report,
            )

        if "HALLUCINATED" not in grounding_report:
            passed_grounding = True
            print("  [grounding] All rules grounded in theory.")
            break

        print("  [grounding] Hallucinated rules found. Repairing extraction...")
        facts_and_rules = reasoning_client.complete(
            RULE_REPAIR_PROMPT.format(
                theory=theory,
                facts_and_rules=facts_and_rules,
                error_report=grounding_report,
            )
        )
        print(facts_and_rules)

        if output_dir:
            _write(
                os.path.join(output_dir, f"facts_and_rules_attempt_{attempt}.txt"),
                facts_and_rules,
            )
    else:
        print(f"  [warning] Max retries reached. Proceeding with best available rules.")

    if output_dir:
        _write(os.path.join(output_dir, "facts_and_rules_final.txt"), facts_and_rules)

    # ── Stage 3: Generate constrained proof ────────────────────────────────────
    print("  [stage 3] Generating constrained proof...")
    proof = reasoning_client.complete(
        CONSTRAINED_PROOF_PROMPT.format(
            facts_and_rules=facts_and_rules,
            question=question,
        )
    )
    print(proof)

    answer = extract_answer(proof)

    if output_dir:
        _write(os.path.join(output_dir, "proof.txt"), proof)
        _write(os.path.join(output_dir, "answer.txt"), answer)

    return {
        "answer": answer,
        "facts_and_rules": facts_and_rules,
        "proof": proof,
        "rule_attempts": rule_attempts,
        "passed_grounding": passed_grounding,
        "grounding_reports": grounding_reports,
    }


# ── Benchmark ────────────────────────────────────────────────────────────────────

def run_benchmark(
    items: list[dict],
    output_dir: str,
    reasoning_client,
    verifier_client,
    max_retries: int = 3,
) -> dict:
    total = len(items)
    correct = 0
    passed_grounding_count = 0
    total_attempts = 0
    results = []

    class_correct = {"True": 0, "False": 0, "Unknown": 0}
    class_total   = {"True": 0, "False": 0, "Unknown": 0}

    print(f"\n{'ID':<22} {'Expected':<10} {'Predicted':<10} {'Grounded':<10} {'OK'}")
    print("-" * 60)

    for item in items:
        item_id  = item["id"]
        theory   = item["theory"]
        question = item["question"]
        expected = item["expected"]
        item_dir = os.path.join(output_dir, item_id)

        print(f"\n[{item_id}] {question[:70]}")
        print(f"  Expected: {expected}")

        result = run_item(
            theory=theory,
            question=question,
            reasoning_client=reasoning_client,
            verifier_client=verifier_client,
            max_retries=max_retries,
            output_dir=item_dir,
        )

        predicted = result["answer"]
        ok = predicted == expected

        if ok:
            correct += 1
        if result["passed_grounding"]:
            passed_grounding_count += 1
        total_attempts += result["rule_attempts"]
        class_total[expected]  = class_total.get(expected, 0) + 1
        if ok:
            class_correct[expected] = class_correct.get(expected, 0) + 1

        results.append({
            "id":               item_id,
            "theory":           theory,
            "question":         question,
            "expected":         expected,
            "predicted":        predicted,
            "correct":          ok,
            "rule_attempts":    result["rule_attempts"],
            "passed_grounding": result["passed_grounding"],
        })

        g_mark  = "✓" if result["passed_grounding"] else "✗"
        ok_mark = "✓" if ok else "✗"
        print(f"\n{item_id:<22} {expected:<10} {predicted:<10} {g_mark:<10} {ok_mark}")

    # ── Aggregate summary ──────────────────────────────────────────────────────
    acc            = correct / total if total else 0
    grounding_rate = passed_grounding_count / total if total else 0
    avg_attempts   = total_attempts / total if total else 0

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
        "total":                      total,
        "correct":                    correct,
        "accuracy":                   round(acc, 4),
        "grounding_rate":             round(grounding_rate, 4),
        "avg_rule_repair_attempts":   round(avg_attempts, 2),
        "per_class":                  per_class,
    }

    print("\n" + "=" * 60)
    print(f"{'PROOFWRITER RESULTS':^60}")
    print("=" * 60)
    print(f"  Total items               : {total}")
    print(f"  Correct                   : {correct}/{total}  ({acc:.1%})")
    print(f"  Rules fully grounded      : {passed_grounding_count}/{total}  ({grounding_rate:.1%})")
    print(f"  Avg rule-repair attempts  : {avg_attempts:.2f}")
    print()
    print(f"  -- Per-Class Accuracy --")
    for cls in ("True", "False", "Unknown"):
        n = class_correct.get(cls, 0)
        d = class_total.get(cls, 0)
        print(f"  {cls:<8}: {n}/{d}  ({pct(n, d)})")
    print("=" * 60)

    _write(os.path.join(output_dir, "summary.json"), json.dumps(summary, indent=2))
    _write(os.path.join(output_dir, "results.json"), json.dumps(results, indent=2))
    print(f"\nWrote {os.path.join(output_dir, 'summary.json')}")
    print(f"Wrote {os.path.join(output_dir, 'results.json')}")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ruleset-constrained proof pipeline on ProofWriter.",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split: train / validation / test (default: test).",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=50,
        help="Number of items to fetch from HuggingFace (default: 50).",
    )
    parser.add_argument(
        "--dataset-file", default=None,
        help=(
            "Use a local JSON file instead of fetching from HuggingFace. "
            "Expects a list of {id, theory, question, expected} objects, "
            "or a dict with an 'items' key."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Root output directory (default: proofwriter_outputs/).",
    )
    parser.add_argument(
        "--max-retries", "-r", type=int, default=3,
        help="Max rule-grounding repair attempts per item (default: 3).",
    )
    parser.add_argument(
        "--client", "-c", choices=["openai", "kimi", "deepseek"], default="openai",
        help=(
            "Reasoning client for theory extraction and proof generation "
            "(default: openai). Verification always uses openai."
        ),
    )
    args = parser.parse_args()

    reasoning_client = make_client(args.client)
    verifier_client  = OpenAILLMClient()

    if args.dataset_file:
        with open(args.dataset_file) as f:
            raw = json.load(f)
        items = raw if isinstance(raw, list) else raw.get("items", raw)
        print(f"Loaded {len(items)} items from {args.dataset_file}")
    else:
        items = fetch_proofwriter(split=args.split, limit=args.limit)

    run_id     = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_dir   = args.output_dir or os.path.join(SCRIPT_DIR, "proofwriter_outputs")
    output_dir = os.path.join(base_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== ProofWriter Benchmark (rule_pipeline) ===")
    print(f"Reasoning client : {args.client}")
    print(f"Verifier client  : openai")
    print(f"Split            : {args.split}")
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
