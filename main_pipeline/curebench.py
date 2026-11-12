"""
CureBench task for the main pipeline: multiple-choice questions built from FDA drug labels.

The pipeline loop (main.run_pipeline / verify_benchmark.run_verify_only) is unchanged; this module
only supplies the CureBench prompts, the letter-answer extraction, and a fact/rule check that
looks evidence up with tools through a CharmChat agent (clients.CharmChatClient):

  - A FACT is SUPPORTED if the question states it, or a tool result confirms it.
  - A RULE is PLAUSIBLE if it is a purely logical principle, or a tool result backs it.
  - Anything with no evidence either way is UNVERIFIED, separately from CONTRADICTED.

The agent cites evidence in CharmChat's <provenance> block, and CharmChat checks each quoted
excerpt against the real tool output. A verdict that relies on tool evidence counts only if a
provenance claim with a verified excerpt backs it; otherwise it is downgraded to UNVERIFIED.
The report quotes the tool output, so the CoT repair step sees the actual evidence.

Report lines keep verify.py's vocabulary (SUPPORTED / UNSUPPORTED, PLAUSIBLE / IMPLAUSIBLE), with
UNVERIFIED or CONTRADICTED as the first word of the reason, so derivation_tree.py parses them.
"""

import json
import re

from main_pipeline.main import (
    Task, STRUCTURED_FOL_TEMPLATE, COT_REPAIR_TEMPLATE, FOL_REPAIR_TEMPLATE,
)
from main_pipeline.clients import CharmChatClient


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old[:60]!r} in the base template")
    return text.replace(old, new)


# ─── Questions and answers ────────────────────────────────────────────────────

def format_question(item: dict) -> str:
    """Question text plus lettered options, as formatQuestion in CharmChat's run-curebench.ts."""
    text = item["question"]
    if item.get("options"):
        text += "\n\nOptions:"
        for key, value in item["options"].items():
            text += f"\n{key}: {value}"
    return text


def extract_answer(text: str) -> str:
    """Letter from the last 'ANSWER: X' in the text, or '' if there is none."""
    matches = re.findall(r'(?i:answer)\s*:\s*\**\s*\(?([A-Z])\b', text)
    return matches[-1] if matches else ""


# ─── Prompts ──────────────────────────────────────────────────────────────────

COT_PROMPT = '''\
I will give you a multiple-choice question about drugs and their use. Provide a logical argument for the answer. Explain every step of your logical reasoning in the format of a logical syllogism where each premise follows from the previous one, with enough context where I can understand your logic even without the full question.

State each fact you rely on explicitly, including what a drug's label says about its indications, contraindications, dosing, warnings, or interactions, and say which option each fact supports or rules out.

End your response with ANSWER: X where X is the letter of your chosen option, with no other text after it. So your formatting should be:
EXPLANATION
ANSWER: X

Here is the question:

{question}
'''

_FACTS_SECTION = '''\
## FACTS
Ground atomic formulas taken directly from the question. No quantifiers.
Each fact is one atomic predicate applied to constants.
A fact may only contain numbers that appear in the question (or direct restatements of them, such as
"a dozen" → 12). Never put a computed value or an arithmetic relationship in a fact.
'''

_CUREBENCH_FACTS_SECTION = '''\
## FACTS
Ground formulas the reasoning takes as given. No quantifiers.
Each fact is one atomic predicate applied to constants, or the negation of one.
A fact either comes from the question (including what each answer option says) or is a specific piece of
drug knowledge the reasoning relies on, such as what a drug's label says about its indications,
contraindications, dosing, warnings, or interactions. Write each such claim as its own fact so that it can
be checked against the label. Never put a computed value or an arithmetic relationship in a fact.
'''

_DRUG_LABEL_EXAMPLE = '''\
=== EXAMPLE 3 (drug label, multiple choice) ===
For a multiple-choice question, the final inference states which option is the answer, e.g. Answer(option_b).

HERE IS EXAMPLE COT REASONING:
The question asks which option is a contraindication for isotretinoin. Isotretinoin's label says it is
contraindicated in pregnancy. Option B is pregnancy, so option B describes a contraindication of isotretinoin.
Option A is severe recalcitrant nodular acne, which isotretinoin is indicated to treat, and a condition a drug
treats is not a contraindication for it, so option A is not the answer. So the answer is B.

CORRECT OUTPUT LOOKS LIKE THIS:

## VOCABULARY
Predicates:
Contraindicated(d, c) — drug d is contraindicated in patients with condition c
Treats(d, c) — drug d is indicated to treat condition c
Describes(o, c) — answer option o describes condition c
Answer(o) — option o is the correct answer
Constants:
isotretinoin, pregnancy, nodular_acne, option_a, option_b

## FACTS
FACT-1: Option A is severe recalcitrant nodular acne :: Describes(option_a, nodular_acne)
FACT-2: Option B is pregnancy :: Describes(option_b, pregnancy)
FACT-3: Isotretinoin is indicated for severe recalcitrant nodular acne :: Treats(isotretinoin, nodular_acne)
FACT-4: Isotretinoin is contraindicated in pregnancy :: Contraindicated(isotretinoin, pregnancy)

## OPERATIONS

## RULES
RULE-1: A drug is not contraindicated in a condition it is indicated to treat :: ∀d∀c (Treats(d, c) → ¬Contraindicated(d, c))
RULE-2: The question asks for a contraindication of isotretinoin, so an option describing one is the answer :: ∀o∀c ((Describes(o, c) ∧ Contraindicated(isotretinoin, c)) → Answer(o))
RULE-3: An option describing a condition isotretinoin is not contraindicated in is not the answer :: ∀o∀c ((Describes(o, c) ∧ ¬Contraindicated(isotretinoin, c)) → ¬Answer(o))

## INFERENCES
INF-1: Isotretinoin is not contraindicated in nodular acne, from FACT-3 and RULE-1 by Universal Instantiation and Modus Ponens :: ¬Contraindicated(isotretinoin, nodular_acne)
INF-2: Option A is not the answer, from FACT-1, INF-1, and RULE-3 by Universal Instantiation and Modus Ponens :: ¬Answer(option_a)
INF-3: Option B is the answer, from FACT-2, FACT-4, and RULE-2 by Universal Instantiation and Modus Ponens :: Answer(option_b)

'''

FOL_TEMPLATE = _replace_once(STRUCTURED_FOL_TEMPLATE, _FACTS_SECTION, _CUREBENCH_FACTS_SECTION)
FOL_TEMPLATE = _replace_once(
    FOL_TEMPLATE,
    "=== CHAIN OF THOUGHT REASONING TO CONVERT ===",
    _DRUG_LABEL_EXAMPLE + "=== CHAIN OF THOUGHT REASONING TO CONVERT ===",
)

CUREBENCH_COT_REPAIR_TEMPLATE = _replace_once(
    COT_REPAIR_TEMPLATE,
    "- Every fact you state is directly supported by the question.",
    "- Every fact you state is supported by the question or by the tool evidence quoted in the error report.\n"
    "- Facts the error report marks CONTRADICTED are replaced with what the evidence actually says.\n"
    "- Facts or rules marked UNVERIFIED are dropped, or restated so they match a drug label.",
)
CUREBENCH_COT_REPAIR_TEMPLATE = _replace_once(
    CUREBENCH_COT_REPAIR_TEMPLATE,
    "EXPLANATION\nThe answer is: X",
    "EXPLANATION\nANSWER: X\nwhere X is the letter of your chosen option.",
)

CUREBENCH_FOL_REPAIR_TEMPLATE = _replace_once(
    FOL_REPAIR_TEMPLATE,
    "- Every FACT is directly supported by or obviously entailed from the story, and contains no calculation.",
    "- Every FACT is stated in the question or is a specific drug fact the reasoning relies on, and contains no "
    "calculation.",
)

VERIFY_PROMPT = '''\
You are checking the premises of a chain of thought that answers a multiple-choice question about drugs.
You are given the question, the FACTS the reasoning takes as given, and the RULES (general principles)
it applies. Use your tools to look up the evidence: start with FDA drug label lookups (contraindications,
indications, dosing, interactions, warnings), and use other sources such as DrugBank or the literature when
the label does not cover it. Judge only from the question and tool results, never from your own background
knowledge. Whether the inferences follow from these premises is checked separately by a theorem prover.

=== QUESTION ===
{story}

=== FACTS ===
{facts}

=== RULES ===
{rules}

For each FACT, write exactly one line, choosing one of:
FACT-N: SUPPORTED (question)
    the question or its options state it (paraphrases count). Use this for what an option says.
FACT-N: SUPPORTED [ev-XXXXXX]
    a tool result states it.
FACT-N: CONTRADICTED [ev-XXXXXX] — <what the evidence says instead>
    a tool result says otherwise. Use (question) instead of the key if the question contradicts it.
FACT-N: UNVERIFIED — <what you searched for>
    you found no evidence either way.

For each RULE, write exactly one line, choosing one of:
RULE-N: PLAUSIBLE (logic)
    a purely logical or definitional principle, or a reading of what the question asks, that needs no
    outside source (e.g. "an option naming a contraindication is the answer to a question asking for one").
RULE-N: PLAUSIBLE [ev-XXXXXX]
    a medical or pharmacological principle that a tool result backs.
RULE-N: IMPLAUSIBLE [ev-XXXXXX] — <why>    or    RULE-N: IMPLAUSIBLE (logic) — <why>
    the evidence contradicts it, or it is logically wrong (inverts a relationship, overgeneralizes).
RULE-N: UNVERIFIED — <what you searched for>
    a medical principle you could not find a source for.

Then write the <provenance> block. Include one claim for every verdict that cites an [ev-XXXXXX] key:
- "claim" must start with the label, e.g. "FACT-2: Isotretinoin is contraindicated in pregnancy".
- "excerpt" must be copied character for character from that tool result: one contiguous span of one or
  two sentences with no line breaks, no ellipses, and no paraphrasing. It is checked by exact string match,
  and a verdict whose excerpt does not match is discarded.

Output ONLY the verdict lines and the <provenance> block.'''


# ─── Fact/rule check ──────────────────────────────────────────────────────────

_VERDICT_RE = re.compile(
    r'^((?:FACT|RULE)-\d+)\s*:\s*'
    r'(SUPPORTED|UNSUPPORTED|PLAUSIBLE|IMPLAUSIBLE|CONTRADICTED|UNVERIFIED)\b(.*)$'
)
_EV_KEY_RE = re.compile(r'ev-[0-9a-f]{6}')
_EVIDENCE_WINDOW = 600


def _parse_verdicts(text: str) -> dict:
    verdicts = {}
    for raw in text.splitlines():
        line = re.sub(r'^[\s\-*•>#]+', '', raw).replace("**", "").strip()
        m = _VERDICT_RE.match(line)
        if not m or m.group(1) in verdicts:
            continue
        label, word, rest = m.groups()
        source = re.search(r'\((question|logic)\)', rest, re.IGNORECASE)
        note = re.split(r'\s[—–-]{1,2}\s', rest, maxsplit=1)
        verdicts[label] = {
            "word": {"UNSUPPORTED": "UNVERIFIED"}.get(word, word),
            "source": source.group(1).lower() if source else None,
            "keys": _EV_KEY_RE.findall(rest),
            "note": note[1].strip() if len(note) > 1 else "",
        }
    return verdicts


def _claim_label(claim: dict):
    m = re.match(r'\W*((?:FACT|RULE)-\d+)', claim.get("claim", ""))
    return m.group(1) if m else None


def _verified_claims(label: str, keys: list, claims: list) -> list:
    """Provenance claims for this label whose excerpt CharmChat found in the cited tool output."""
    return [
        c for c in claims
        if c.get("keyValid") and c.get("excerptVerified")
        and (_claim_label(c) == label or (_claim_label(c) is None and c.get("evidenceKey") in keys))
    ]


def _judge(label: str, is_fact: bool, verdict, claims: list):
    """Return (report line, has_error, verified claims used)."""
    ok_word, bad_word = ("SUPPORTED", "UNSUPPORTED") if is_fact else ("PLAUSIBLE", "IMPLAUSIBLE")
    if verdict is None:
        return f"{label}: {bad_word} — UNVERIFIED: the verifier gave no verdict", True, []

    word, source, note = verdict["word"], verdict["source"], verdict["note"]
    backing = _verified_claims(label, verdict["keys"], claims)
    cited = ", ".join(sorted({c["evidenceKey"] for c in backing}))

    if word in ("SUPPORTED", "PLAUSIBLE"):
        if source:
            reason = "stated in the question" if source == "question" else "logical principle, no source needed"
            return f"{label}: {ok_word} — {reason}", False, []
        if backing:
            return f"{label}: {ok_word} — confirmed by tool evidence {cited}", False, backing
        return (f"{label}: {bad_word} — UNVERIFIED: marked {word} but no cited tool output contains the "
                f"quoted evidence", True, [])

    if word in ("CONTRADICTED", "IMPLAUSIBLE"):
        detail = note or "contradicted"
        if backing:
            return f"{label}: {bad_word} — CONTRADICTED by tool evidence {cited}: {detail}", True, backing
        if source:
            return f"{label}: {bad_word} — CONTRADICTED by the {source}: {detail}", True, []
        return (f"{label}: {bad_word} — UNVERIFIED: marked {word} ({detail}) but no cited tool output "
                f"contains the quoted evidence", True, [])

    return f"{label}: {bad_word} — UNVERIFIED: {note or 'no evidence found'}", True, []


def _evidence_block(label: str, claim: dict, evidence_store: dict) -> str:
    entry = evidence_store.get(claim["evidenceKey"], {})
    content = entry.get("content", "")
    excerpt = claim.get("excerpt", "")
    start = content.lower().find(excerpt.lower()) if excerpt else -1
    if start < 0:
        window = content[:2 * _EVIDENCE_WINDOW]
    else:
        lo = max(0, start - _EVIDENCE_WINDOW)
        hi = min(len(content), start + len(excerpt) + _EVIDENCE_WINDOW)
        window = ("…" if lo else "") + content[lo:hi] + ("…" if hi < len(content) else "")
    tool = entry.get("toolName", "?").split("__")[-1]
    return (
        f"{label} [{claim['evidenceKey']}] {tool} {json.dumps(entry.get('args', {}))}\n"
        f"  quote: \"{excerpt}\"\n"
        f"  tool output: {' '.join(window.split())}"
    )


def verify_fol(story_text: str, structured_fol: str, client=None) -> tuple[bool, str]:
    """
    Tool-backed fact grounding + rule plausibility check, run through a CharmChat agent.
    Same signature and return value as verify.verify_fol: (has_errors, report_text).
    """
    if not isinstance(client, CharmChatClient):
        raise TypeError("CureBench verification needs a CharmChatClient as the verifier client.")

    facts = re.findall(r'^(FACT-\d+):\s*(.+?)\s*::\s*(.+)$', structured_fol, re.MULTILINE)
    rules = re.findall(r'^(RULE-\d+):\s*(.+?)\s*::\s*(.+)$', structured_fol, re.MULTILINE)
    if not facts and not rules:
        return True, "No FACT or RULE lines found in the structured FOL."

    response = client.complete(VERIFY_PROMPT.format(
        story=story_text,
        facts="\n".join(f"{label}: {gloss}" for label, gloss, _ in facts),
        rules="\n".join(f"{label}: {gloss}" for label, gloss, _ in rules),
    ))
    call = client.last_call
    provenance = call.get("provenance") or {}
    claims = provenance.get("claims", [])
    evidence_store = provenance.get("evidenceStore", {})
    verdicts = _parse_verdicts(response)

    lines, evidence, has_errors = [], [], False
    for items, is_fact in ((facts, True), (rules, False)):
        for label, _, _ in items:
            line, error, backing = _judge(label, is_fact, verdicts.get(label), claims)
            lines.append(line)
            has_errors |= error
            evidence += [_evidence_block(label, c, evidence_store) for c in backing]

    tool_calls = call.get("tool_calls", [])
    empty = sum(1 for e in evidence_store.values() if e.get("isEmpty"))
    unverified_quotes = sum(1 for c in claims if not c.get("excerptVerified"))
    report = "\n".join(lines)
    report += (f"\n\nTool calls: {len(tool_calls)} ({empty} returned no data). "
               f"Provenance claims: {len(claims)} ({unverified_quotes} with quotes not found in the tool output).")
    if call.get("errors"):
        report += "\nVerifier errors: " + "; ".join(call["errors"])
    if evidence:
        report += "\n\n=== EVIDENCE FROM TOOL CALLS ===\n" + "\n\n".join(evidence)
    return has_errors, report


CUREBENCH_TASK = Task(
    cot_prompt=COT_PROMPT,
    fol_template=FOL_TEMPLATE,
    cot_repair_template=CUREBENCH_COT_REPAIR_TEMPLATE,
    fol_repair_template=CUREBENCH_FOL_REPAIR_TEMPLATE,
    extract_answer=extract_answer,
    verify_fol=verify_fol,
)
