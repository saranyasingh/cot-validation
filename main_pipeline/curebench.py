"""
CureBench task for the main pipeline: multiple-choice questions built from FDA drug labels.

The pipeline loop (main.run_pipeline / verify_benchmark.run_verify_only) is unchanged; this module
only supplies the CureBench prompts, the letter-answer extraction, and a fact/rule check that
looks evidence up with tools through a CharmChat agent (clients.CharmChatClient):

  - A FACT is SUPPORTED if the question states it, or a tool result confirms it.
  - A RULE is PLAUSIBLE if it is a purely logical principle, or a tool result backs it.
  - Anything with no evidence either way is UNVERIFIED, separately from CONTRADICTED.

The agent cites evidence in CharmChat's <provenance> block. Each quoted excerpt is checked against
the real tool outputs here (check_claims), tolerating JSON escapes, whitespace, and punctuation
differences. A verdict that relies on tool evidence counts only if a provenance claim with a
verified excerpt backs it; otherwise it is downgraded to UNVERIFIED.
The report quotes the tool output, so the CoT repair step sees the actual evidence.

The agent is told to make at most MAX_TOOL_CALLS tool calls. If the call fails (an error such as
CharmChat's iteration limit, an exception, or no verdict lines), it is retried once with a tighter
budget, and the attempt with more verdicts is used.

Report lines keep verify.py's vocabulary (SUPPORTED / UNSUPPORTED, PLAUSIBLE / IMPLAUSIBLE), with
UNVERIFIED or CONTRADICTED as the first word of the reason, so derivation_tree.py parses them.
"""

import json
import re
import unicodedata

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

TOOL BUDGET: make at most {max_tool_calls} tool calls in total, then stop searching and write your verdicts.
- Prefer the per-section FDA label tools (e.g. FDA_get_adverse_reactions_by_drug_name,
  FDA_get_warnings_and_cautions_by_drug_name, FDA_get_indications_by_drug_name); use FDA_get_drug_label
  only when you need several sections of one label.
- One lookup can cover several facts and rules about the same drug. Never repeat a lookup, and if a drug
  name returns nothing, try at most one alternative name (brand or generic).
- When the budget is spent, mark whatever you could not check UNVERIFIED. Verdicts for every FACT and
  RULE are required; running out of tool calls without writing them wastes all the evidence.
{retry_note}
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
  - "excerpt" must be copied word for word from that tool result: one contiguous span of one or two
  sentences (at least 5 words) with no ellipses and no paraphrasing. It is checked against the tool
  output, and a verdict whose excerpt is not found there is discarded.

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


# ─── Quote matching ───────────────────────────────────────────────────────────
#
# CharmChat's own check (exact lowercase substring of the raw tool output) misses quotes the agent
# copied faithfully: tool results are often JSON, so the raw text has \" and \n escapes and the
# agent's quote does not; whitespace, curly quotes, markdown, and HTML tags differ; and one JSON
# error in the <provenance> block drops every claim. Claims are re-parsed from the raw response
# and re-checked against the tool outputs here.

_SHINGLE = 4                # words per shingle for the near-exact match
_MIN_SHINGLE_COVERAGE = 0.85
_MIN_QUOTE_WORDS = 5        # shorter quotes ("indicated for the treatment") match almost any label
_CHAR_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-",
                           "\u00a0": " ", "*": " ", "`": " ", "#": " "})


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    text = re.sub(r'</?[a-z][a-z0-9]*(?:\s[^<>]{0,300})?/?>', ' ', text, flags=re.IGNORECASE)
    return " ".join(text.lower().split())


def _words(normalized: str) -> list:
    return re.findall(r'[a-z0-9]+(?:[.,][0-9]+)*', normalized)


def _json_strings(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [str(k)] + _json_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _json_strings(v)]
    return [str(value)] if value is not None else []


def _searchable(content: str) -> dict:
    """The tool output as normalized text and words, with JSON string escapes undone."""
    variants = [content]
    try:
        variants.append(" \n ".join(_json_strings(json.loads(content))))
    except (json.JSONDecodeError, TypeError, ValueError):
        unescaped = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), content)
        for old, new in (('\\n', '\n'), ('\\t', ' '), ('\\"', '"'), ('\\/', '/'), ('\\\\', '\\')):
            unescaped = unescaped.replace(old, new)
        variants.append(unescaped)
    text = " | ".join(_normalize(v) for v in variants)
    words = _words(text)
    shingles = {tuple(words[i:i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}
    return {"text": text, "display": _normalize(variants[-1]),
            "words": " " + " ".join(words) + " ", "shingles": shingles}


def _match_fragment(fragment: str, searchable: dict):
    """'exact', 'normalized', or 'near-exact (NN%)' if the fragment is in the tool output, else None."""
    norm = _normalize(fragment)
    words = _words(norm)
    if len(words) < _MIN_QUOTE_WORDS:
        return None
    if norm in searchable["text"]:
        return "exact"
    if " " + " ".join(words) + " " in searchable["words"]:
        return "normalized"
    if len(words) >= _SHINGLE + 2:
        shingles = [tuple(words[i:i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)]
        coverage = sum(s in searchable["shingles"] for s in shingles) / len(shingles)
        if coverage >= _MIN_SHINGLE_COVERAGE:
            return f"near-exact ({coverage:.0%})"
    return None


def match_excerpt(excerpt: str, searchable: dict):
    """Match a quote against one tool output. Quotes the agent split with ellipses must match
    piece by piece. Returns the weakest match kind, or None."""
    fragments = [f for f in re.split(r'\.{3,}|…', excerpt.strip().strip('"\'')) if len(_words(_normalize(f))) >= _MIN_QUOTE_WORDS]
    if not fragments:
        return None
    kinds = [_match_fragment(f, searchable) for f in fragments]
    if any(k is None for k in kinds):
        return None
    for kind in ("near-exact", "normalized", "exact"):
        hits = [k for k in kinds if k.startswith(kind)]
        if hits:
            return hits[0]


def parse_provenance(raw_text: str) -> list:
    """Claims from the last <provenance> block. Falls back to pulling out each claim object on its
    own when the block as a whole is not valid JSON."""
    blocks = re.findall(r'<provenance>([\s\S]*?)(?:</provenance>|$)', raw_text or "", re.IGNORECASE)
    if not blocks:
        return []
    block = blocks[-1].strip()
    block = re.sub(r'^```(?:json)?|```$', '', block).strip()
    try:
        parsed = json.loads(block)
        if isinstance(parsed, list):
            return [c for c in parsed if isinstance(c, dict) and isinstance(c.get("excerpt"), str)]
    except json.JSONDecodeError:
        pass
    claims = []
    for obj in re.findall(r'\{[^{}]*\}', block):
        fields = {}
        for name in ("claim", "evidenceKey", "excerpt"):
            m = re.search(rf'"{name}"\s*:\s*"((?:[^"\\]|\\.|"(?!\s*[,}}]))*)"', obj, re.DOTALL)
            if m:
                fields[name] = m.group(1).replace('\\"', '"').replace('\\n', ' ')
        if "excerpt" in fields:
            claims.append(fields)
    return claims


def check_claims(call: dict) -> tuple[list, dict]:
    """Re-check every provenance claim from a CharmChatClient call against the tool outputs.

    A claim's key is valid if it names a tool output that returned data. Its excerpt is verified if
    it matches the cited output, or else any other output from the same call (the agent sometimes
    cites the wrong key); `matchedKey` and `match` record where and how. Returns (claims, evidence).
    """
    provenance = call.get("provenance") or {}
    evidence = {**call.get("evidence", {}), **provenance.get("evidenceStore", {})}
    usable = {k: e for k, e in evidence.items() if not e.get("isEmpty") and e.get("content")}
    searchable = {}

    def lookup(key):
        if key not in searchable:
            searchable[key] = _searchable(usable[key]["content"])
        return searchable[key]

    raw_claims = parse_provenance(call.get("raw_text", "")) or provenance.get("claims", [])
    checked = []
    for raw in raw_claims:
        claim = {
            "claim": str(raw.get("claim", "")),
            "evidenceKey": str(raw.get("evidenceKey", "")),
            "excerpt": str(raw.get("excerpt", "")),
            "keyValid": raw.get("evidenceKey") in usable,
            "excerptVerified": False,
            "matchedKey": None,
            "match": None,
        }
        order = [claim["evidenceKey"]] if claim["keyValid"] else []
        order += [k for k in usable if k != claim["evidenceKey"]]
        for key in order:
            kind = match_excerpt(claim["excerpt"], lookup(key))
            if kind:
                where = "" if key == claim["evidenceKey"] else \
                    f", found in {key} instead of cited {claim['evidenceKey'] or '(no key)'}"
                claim.update(excerptVerified=True, matchedKey=key, match=f"{kind} match{where}")
                break
        checked.append(claim)
    return checked, evidence


def _verified_claims(label: str, keys: list, claims: list) -> list:
    """Provenance claims for this label whose excerpt was found in a tool output."""
    return [
        c for c in claims
        if c.get("excerptVerified")
        and (_claim_label(c) == label
             or (_claim_label(c) is None and {c.get("evidenceKey"), c.get("matchedKey")} & set(keys)))
    ]


def _judge(label: str, is_fact: bool, verdict, claims: list):
    """Return (report line, has_error, verified claims used)."""
    ok_word, bad_word = ("SUPPORTED", "UNSUPPORTED") if is_fact else ("PLAUSIBLE", "IMPLAUSIBLE")
    if verdict is None:
        return f"{label}: {bad_word} — UNVERIFIED: the verifier gave no verdict", True, []

    word, source, note = verdict["word"], verdict["source"], verdict["note"]
    backing = _verified_claims(label, verdict["keys"], claims)
    cited = ", ".join(sorted({c["matchedKey"] for c in backing}))

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
    entry = evidence_store.get(claim["matchedKey"], {})
    excerpt = claim.get("excerpt", "")
    text = _searchable(entry.get("content", ""))["display"]
    anchor = _normalize(re.split(r'\.{3,}|…', excerpt.strip().strip('"\''))[0])
    start = text.find(anchor[:80]) if anchor else -1
    if start < 0:
        window = text[:2 * _EVIDENCE_WINDOW]
    else:
        lo = max(0, start - _EVIDENCE_WINDOW)
        hi = min(len(text), start + len(anchor) + _EVIDENCE_WINDOW)
        window = ("…" if lo else "") + text[lo:hi] + ("…" if hi < len(text) else "")
    tool = entry.get("toolName", "?").split("__")[-1]
    return (
        f"{label} [{claim['matchedKey']}] {tool} {json.dumps(entry.get('args', {}))}\n"
        f"  quote ({claim['match']}): \"{excerpt}\"\n"
        f"  tool output: {window}"
    )


MAX_TOOL_CALLS = 6
RETRY_MAX_TOOL_CALLS = 4
RETRY_NOTE = '''
NOTE: a previous attempt at this check failed ({failure}). This time make at most {max_tool_calls} tool
calls, the most informative ones first, and write all your verdicts even if some stay UNVERIFIED.
'''


def _verify_attempt(client, prompt_args: dict, max_tool_calls: int, retry_note: str) -> dict:
    """One verifier call. `failure` says why it produced no usable verdicts (None if it did)."""
    prompt = VERIFY_PROMPT.format(max_tool_calls=max_tool_calls, retry_note=retry_note, **prompt_args)
    try:
        response = client.complete(prompt)
        call = client.last_call
    except Exception as e:
        print(f"WARNING: CharmChat verifier call failed: {e}")
        response, call = "", {"errors": [f"{type(e).__name__}: {e}"]}
    verdicts = _parse_verdicts(response)
    failure = None
    if call.get("errors"):
        failure = "; ".join(call["errors"])
    elif not verdicts:
        failure = f"no verdict lines in the response after {len(call.get('tool_calls', []))} tool calls"
    return {"call": call, "verdicts": verdicts, "failure": failure}


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

    prompt_args = dict(
        story=story_text,
        facts="\n".join(f"{label}: {gloss}" for label, gloss, _ in facts),
        rules="\n".join(f"{label}: {gloss}" for label, gloss, _ in rules),
    )
    attempts = [_verify_attempt(client, prompt_args, MAX_TOOL_CALLS, "")]
    if attempts[0]["failure"]:
        retry_note = RETRY_NOTE.format(failure=attempts[0]["failure"], max_tool_calls=RETRY_MAX_TOOL_CALLS)
        attempts.append(_verify_attempt(client, prompt_args, RETRY_MAX_TOOL_CALLS, retry_note))
    best = max(attempts, key=lambda a: (len(a["verdicts"]), not a["failure"]))
    call, verdicts = best["call"], best["verdicts"]
    claims, evidence_store = check_claims(call)

    lines, evidence, has_errors = [], [], False
    for items, is_fact in ((facts, True), (rules, False)):
        for label, _, _ in items:
            line, error, backing = _judge(label, is_fact, verdicts.get(label), claims)
            lines.append(line)
            has_errors |= error
            evidence += [_evidence_block(label, c, evidence_store) for c in backing]

    tool_calls = call.get("tool_calls", [])
    empty = sum(1 for e in evidence_store.values() if e.get("isEmpty"))
    unverified_quotes = [c for c in claims if not c["excerptVerified"]]
    report = "\n".join(lines)
    report += (f"\n\nTool calls: {len(tool_calls)} ({empty} returned no data). "
               f"Provenance claims: {len(claims)} ({len(unverified_quotes)} with quotes not found in any tool output).")
    for c in unverified_quotes:
        report += f"\n  quote not found [{c['evidenceKey'] or 'no key'}]: \"{c['excerpt'][:200]}\""
    if len(attempts) > 1:
        report += (f"\nVerifier retried after: {attempts[0]['failure']}. "
                   f"Retry {'also failed: ' + attempts[1]['failure'] if attempts[1]['failure'] else 'succeeded'}; "
                   f"verdicts above are from attempt {attempts.index(best) + 1}.")
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
