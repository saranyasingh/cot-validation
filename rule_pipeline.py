"""
rule_pipeline.py — Ruleset-constrained proof pipeline on ProofWriter.

Key difference from main.py:
  - RULES are extracted only from the given theory — no LLM-invented rules.
  - Verification checks that every extracted rule is explicitly grounded in the
    theory text before the proof is generated.
  - Tracks pre-verification vs post-verification accuracy to measure how much
    rule grounding actually helps.

Feed data from load_data.py:
    python load_data.py --proofwriter --limit 100
    python rule_pipeline.py --dataset-file proofwriter.json

Usage:
    python rule_pipeline.py --dataset-file proofwriter.json
    python rule_pipeline.py --dataset-file proofwriter.json --client kimi
    python rule_pipeline.py --dataset-file proofwriter.json --output-dir ./pw_run
"""

import argparse
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv

from clients import make_client, OpenAILLMClient

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Prompt Templates ────────────────────────────────────────────────────────────

FACTS_TO_FOL_PROMPT = '''\
Convert each numbered fact below into a ground FOL formula. Translate only — do not add or remove facts.

=== SYNTAX ===
- Predicates: CamelCase (e.g., IsYoung, IsBig, Chases)
- Constants: lowercase with underscores (e.g., bald_eagle, anne, the_cat)
- No quantifiers — these are ground atoms.

=== OUTPUT FORMAT ===
FACT-1: <original sentence> :: <FOL formula>
FACT-2: ...

Output ONLY the labeled lines, one per fact.

=== FACTS ===
{facts}
'''

RULES_TO_FOL_PROMPT = '''\
Convert each numbered rule below into a universally quantified FOL formula. Translate only — do not add or remove rules.

=== SYNTAX ===
- Quantifiers: ∀ (universal)
- Connectives: ∧ (and), ∨ (or), → (implies), ¬ (not)
- Predicates: CamelCase (e.g., IsYoung, IsBig, Chases)
- Variables: single lowercase letters (x, y, z)
- Every rule MUST start with ∀.

=== OUTPUT FORMAT ===
RULE-1: <original sentence> :: <FOL formula>
RULE-2: ...

Output ONLY the labeled lines, one per rule.

=== RULES ===
{rules}
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
You are given a set of FACTS and RULES extracted from a theory, and a statement to evaluate.

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


# ── Utilities ────────────────────────────────────────────────────────────────────

def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def extract_answer(text: str) -> str:
    match = re.search(r'^ANSWER:\s*(True|False|Unknown)', text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).capitalize() if match else "Unknown"


def extract_rules(facts_and_rules: str) -> str:
    lines = [l.strip() for l in facts_and_rules.splitlines() if re.match(r'RULE-\d+:', l.strip())]
    return "\n".join(lines) if lines else "(no rules)"


# ── Per-Item Pipeline ────────────────────────────────────────────────────────────

def run_item(
    facts: list[str],
    rules: list[str],
    question: str,
    reasoning_client,
    verifier_client,
    max_retries: int = 3,
    output_dir: str = None,
) -> dict:
    """
    Constrained proof pipeline for one ProofWriter item.

    Facts and rules are loaded directly from the dataset — the LLM only
    translates them to FOL, it does not extract or invent them.

    Stage 1: Translate pre-loaded facts + rules to FOL (reasoning client)
    Stage 2: Prove with unverified FOL → pre_answer
    Stage 3: Verify FOL translation didn't add rules; repair if so (verifier + reasoning)
    Stage 4: Prove with verified FOL → post_answer

    Returns: {pre_answer, post_answer, facts_and_rules_final, rule_attempts, passed_grounding}
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _write(os.path.join(output_dir, "facts_input.txt"),   "\n".join(facts))
        _write(os.path.join(output_dir, "rules_input.txt"),   "\n".join(rules))
        _write(os.path.join(output_dir, "question.txt"),      question)

    # ── Stage 1: Translate pre-loaded facts + rules to FOL ────────────────────
    print("  [stage 1] Translating dataset facts and rules to FOL...")
    numbered_facts = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    numbered_rules = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules))

    facts_fol = reasoning_client.complete(FACTS_TO_FOL_PROMPT.format(facts=numbered_facts))
    rules_fol = reasoning_client.complete(RULES_TO_FOL_PROMPT.format(rules=numbered_rules))
    facts_and_rules = f"## FACTS\n{facts_fol}\n\n## RULES\n{rules_fol}"
    print(facts_and_rules)
    if output_dir:
        _write(os.path.join(output_dir, "facts_and_rules_raw.txt"), facts_and_rules)

    # ── Stage 2: Prove with raw (unverified) FOL translation ──────────────────
    print("  [stage 2] Generating pre-verification proof...")
    pre_proof = reasoning_client.complete(
        CONSTRAINED_PROOF_PROMPT.format(facts_and_rules=facts_and_rules, question=question)
    )
    pre_answer = extract_answer(pre_proof)
    print(f"  Pre-verification answer: {pre_answer}")
    if output_dir:
        _write(os.path.join(output_dir, "proof_pre.txt"), pre_proof)

    # ── Stage 3: Verify the FOL translation didn't hallucinate extra rules ─────
    # We check the translated rules against the original natural-language rules.
    grounding_reports = []
    passed_grounding = False
    rule_attempts = 0
    theory_rules_text = "\n".join(rules)   # ground truth from dataset

    for attempt in range(1, max_retries + 1):
        rule_attempts = attempt
        rules_text = extract_rules(facts_and_rules)

        print(f"  [stage 3, attempt {attempt}] Checking rule grounding...")
        grounding_report = verifier_client.complete(
            RULE_GROUNDING_PROMPT.format(theory=theory_rules_text, rules=rules_text)
        ).strip()
        grounding_reports.append(grounding_report)
        print(grounding_report)
        if output_dir:
            _write(os.path.join(output_dir, f"grounding_attempt_{attempt}.txt"), grounding_report)

        if "HALLUCINATED" not in grounding_report:
            passed_grounding = True
            print("  [grounding] All rules match dataset.")
            break

        print("  [grounding] Extra rules found in translation. Repairing...")
        rules_fol = reasoning_client.complete(
            RULE_REPAIR_PROMPT.format(
                theory=theory_rules_text,
                facts_and_rules=facts_and_rules,
                error_report=grounding_report,
            )
        )
        facts_and_rules = f"## FACTS\n{facts_fol}\n\n## RULES\n{rules_fol}"
        print(facts_and_rules)
        if output_dir:
            _write(os.path.join(output_dir, f"facts_and_rules_attempt_{attempt}.txt"), facts_and_rules)
    else:
        print("  [warning] Max retries reached. Proceeding with best available translation.")

    if output_dir:
        _write(os.path.join(output_dir, "facts_and_rules_final.txt"), facts_and_rules)

    # ── Stage 4: Prove with verified FOL ──────────────────────────────────────
    print("  [stage 4] Generating post-verification proof...")
    post_proof = reasoning_client.complete(
        CONSTRAINED_PROOF_PROMPT.format(facts_and_rules=facts_and_rules, question=question)
    )
    post_answer = extract_answer(post_proof)
    print(f"  Post-verification answer: {post_answer}")
    if output_dir:
        _write(os.path.join(output_dir, "proof_post.txt"), post_proof)

    return {
        "pre_answer":        pre_answer,
        "post_answer":       post_answer,
        "facts_and_rules":   facts_and_rules,
        "rule_attempts":     rule_attempts,
        "passed_grounding":  passed_grounding,
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
    pre_correct = post_correct = 0
    passed_grounding_count = total_attempts = 0
    corrected = 0   # wrong pre → right post  (verification helped)
    regressed = 0   # right pre → wrong post  (verification hurt)
    results = []

    pre_class_correct  = {"True": 0, "False": 0, "Unknown": 0}
    post_class_correct = {"True": 0, "False": 0, "Unknown": 0}
    class_total        = {"True": 0, "False": 0, "Unknown": 0}

    print(f"\n{'ID':<22} {'Expected':<10} {'Pre':>6} {'Post':>6} {'Grnd':>6} {'Fixed':>6}")
    print("-" * 62)

    for item in items:
        item_id  = item["id"]
        question = item["question"]
        expected = item["expected"]
        item_dir = os.path.join(output_dir, item_id)

        facts    = item["facts"]
        rules    = item["rules"]

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

        pre_ans  = result["pre_answer"]
        post_ans = result["post_answer"]
        pre_ok   = pre_ans  == expected
        post_ok  = post_ans == expected

        if pre_ok:
            pre_correct += 1
        if post_ok:
            post_correct += 1
        if result["passed_grounding"]:
            passed_grounding_count += 1
        total_attempts += result["rule_attempts"]

        class_total[expected]         = class_total.get(expected, 0) + 1
        if pre_ok:
            pre_class_correct[expected]  = pre_class_correct.get(expected, 0) + 1
        if post_ok:
            post_class_correct[expected] = post_class_correct.get(expected, 0) + 1

        fixed = ""
        if not pre_ok and post_ok:
            corrected += 1
            fixed = "✓ fixed"
        elif pre_ok and not post_ok:
            regressed += 1
            fixed = "✗ regr."

        results.append({
            "id":               item_id,
            "question":         question,
            "expected":         expected,
            "num_facts":        len(facts),
            "num_rules":        len(rules),
            "pre_answer":       pre_ans,
            "post_answer":      post_ans,
            "pre_correct":      pre_ok,
            "post_correct":     post_ok,
            "rule_attempts":    result["rule_attempts"],
            "passed_grounding": result["passed_grounding"],
        })

        pre_m  = "✓" if pre_ok  else "✗"
        post_m = "✓" if post_ok else "✗"
        g_m    = "✓" if result["passed_grounding"] else "✗"
        print(f"\n{item_id:<22} {expected:<10} {pre_m:>6} {post_m:>6} {g_m:>6}  {fixed}")

    # ── Aggregate summary ──────────────────────────────────────────────────────
    pre_acc        = pre_correct  / total if total else 0
    post_acc       = post_correct / total if total else 0
    delta          = post_acc - pre_acc
    grounding_rate = passed_grounding_count / total if total else 0
    avg_attempts   = total_attempts / total if total else 0

    def pct(n, d):
        return f"{n/d:.1%}" if d else "n/a"

    per_class = {
        cls: {
            "total":          class_total.get(cls, 0),
            "pre_correct":    pre_class_correct.get(cls, 0),
            "post_correct":   post_class_correct.get(cls, 0),
            "pre_accuracy":   round(pre_class_correct.get(cls, 0)  / class_total[cls], 4) if class_total.get(cls) else 0,
            "post_accuracy":  round(post_class_correct.get(cls, 0) / class_total[cls], 4) if class_total.get(cls) else 0,
        }
        for cls in ("True", "False", "Unknown")
    }

    summary = {
        "total":                    total,
        "pre_verification":  {"correct": pre_correct,  "accuracy": round(pre_acc,  4)},
        "post_verification": {"correct": post_correct, "accuracy": round(post_acc, 4)},
        "delta":                    round(delta, 4),
        "corrected_by_verification": corrected,
        "regressed_by_verification": regressed,
        "grounding_rate":            round(grounding_rate, 4),
        "avg_rule_repair_attempts":  round(avg_attempts, 2),
        "per_class":                 per_class,
    }

    direction = "better" if delta > 0 else "worse" if delta < 0 else "no change"

    print("\n" + "=" * 62)
    print(f"{'PROOFWRITER RESULTS':^62}")
    print("=" * 62)
    print(f"  Total items                  : {total}")
    print(f"  Pre-verification accuracy    : {pre_correct}/{total}  ({pre_acc:.1%})")
    print(f"  Post-verification accuracy   : {post_correct}/{total}  ({post_acc:.1%})")
    print(f"  Delta                        : {delta:+.1%} ({direction})")
    print()
    print(f"  Corrected by verification    : {corrected}  (wrong→right)")
    print(f"  Regressed by verification    : {regressed}  (right→wrong)")
    print()
    print(f"  Rules fully grounded         : {passed_grounding_count}/{total}  ({grounding_rate:.1%})")
    print(f"  Avg rule-repair attempts     : {avg_attempts:.2f}")
    print()
    print(f"  -- Per-Class Accuracy (pre → post) --")
    for cls in ("True", "False", "Unknown"):
        n_pre  = pre_class_correct.get(cls, 0)
        n_post = post_class_correct.get(cls, 0)
        d      = class_total.get(cls, 0)
        print(f"  {cls:<8}: {pct(n_pre, d)} → {pct(n_post, d)}  ({d} items)")
    print("=" * 62)

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
        "--dataset-file", "-f", required=True,
        help="Path to a ProofWriter JSON file (from load_data.py --proofwriter).",
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

    with open(args.dataset_file) as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get("items", raw)
    print(f"Loaded {len(items)} items from {args.dataset_file}")

    run_id     = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_dir   = args.output_dir or os.path.join(SCRIPT_DIR, "proofwriter_outputs")
    output_dir = os.path.join(base_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== ProofWriter Benchmark (rule_pipeline) ===")
    print(f"Reasoning client : {args.client}")
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
