
import re
from typing import Optional


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_inf_premises(gloss: str) -> tuple[list[str], Optional[str]]:
    """Extract premise labels and inference rule from an inference gloss.

    E.g. "Dividing 6 by 2 gives 3, from FACT-1 and FACT-2 by Arithmetic Computation"
    → (["FACT-1", "FACT-2"], "Arithmetic Computation")
    """
    m = re.search(r'\bfrom\s+(.+?)\s+by\s+(.+)$', gloss, re.IGNORECASE)
    if not m:
        return [], None
    premises = re.findall(r'(?:FACT|RULE|INF)-\d+', m.group(1))
    return premises, m.group(2).strip()


def parse_structured_fol(structured_fol: str) -> dict:
    """Parse structured FOL text into facts, rules, and inferences.

    Returns {"facts": {...}, "rules": {...}, "inferences": {...}} each keyed by label.
    """
    facts: dict = {}
    rules: dict = {}
    inferences: dict = {}

    for raw_line in structured_fol.splitlines():
        line = raw_line.strip().lstrip('#').strip()

        m = re.match(r'^(FACT-\d+):\s*(.+?)\s*::\s*(.+)$', line)
        if m:
            label, gloss, formula = m.group(1), m.group(2), m.group(3)
            facts[label] = {"type": "fact", "gloss": gloss, "formula": formula}
            continue

        m = re.match(r'^(RULE-\d+):\s*(.+?)\s*::\s*(.+)$', line)
        if m:
            label, gloss, formula = m.group(1), m.group(2), m.group(3)
            rules[label] = {"type": "rule", "gloss": gloss, "formula": formula}
            continue

        m = re.match(r'^(INF-\d+):\s*(.+?)\s*::\s*(.+)$', line)
        if m:
            label, gloss, formula = m.group(1), m.group(2), m.group(3)
            premises, inf_rule = _parse_inf_premises(gloss)
            inferences[label] = {
                "type": "inference",
                "gloss": gloss,
                "formula": formula,
                "premises": premises,
                "inference_rule": inf_rule,
            }

    return {"facts": facts, "rules": rules, "inferences": inferences}


def parse_verify_report(verify_report: str) -> dict:
    """Parse verify.py report text into per-label status dicts.

    Returns {"FACT-1": {"status": "SUPPORTED", "reason": None}, ...}
    """
    results: dict = {}
    for raw_line in verify_report.splitlines():
        line = raw_line.strip()
        m = re.match(
            r'^((?:FACT|RULE)-\d+):\s*'
            r'(SUPPORTED|UNSUPPORTED|PLAUSIBLE|IMPLAUSIBLE)'
            r'(?:\s*[—–\-]+\s*(.+))?$',
            line,
        )
        if m:
            label = m.group(1)
            status = m.group(2)
            reason = m.group(3).strip() if m.group(3) else None
            results[label] = {"status": status, "reason": reason}
    return results


def parse_tptp_report(tptp_report: str) -> dict:
    """Parse TPTP prover report into per-inference status dicts.

    Returns {"INF-1": {"status": "PROVED", "reason": None}, ...}

    Handles both summary lines ("inf_1: PROVED") and error-section lines
    ("inf_1: NOT PROVED — reason"), preferring entries that include a reason.
    """
    results: dict = {}
    for raw_line in tptp_report.splitlines():
        line = raw_line.strip()
        m = re.match(
            r'^(inf_\d+):\s*'
            r'(NOT PROVED|PROVED|TIMEOUT|UNKNOWN(?:\s+\([^)]+\))?)'
            r'(?:\s*[—–\-]+\s*(.+))?$',
            line,
            re.IGNORECASE,
        )
        if m:
            inf_key = m.group(1).lower()           # inf_1
            status = m.group(2).upper()
            reason = m.group(3).strip() if m.group(3) else None
            label = re.sub(r'inf_(\d+)', lambda x: f'INF-{x.group(1)}', inf_key)
            existing = results.get(label)
            # Prefer the entry that carries a reason (the "Prover errors" section)
            if existing is None or (reason and not existing["reason"]):
                results[label] = {"status": status, "reason": reason}
    return results


# ── Tree builder ──────────────────────────────────────────────────────────────

def build_derivation_tree(
    structured_fol: str,
    verify_report: str,
    tptp_report: str,
) -> dict:
    """Build a parsable derivation tree annotated with verification errors.

    Schema:
        nodes       — dict keyed by label (FACT-N, RULE-N, INF-N) with full annotation
        root        — label of the final inference (the conclusion)
        errors      — list of {node, stage, message} for every detected error
        error_stages — sorted list of stages that fired ("verify", "tptp")
    """
    parsed = parse_structured_fol(structured_fol)
    verify_statuses = parse_verify_report(verify_report)
    tptp_statuses = parse_tptp_report(tptp_report)

    nodes: dict = {}
    errors: list = []

    for label, data in parsed["facts"].items():
        v = verify_statuses.get(label, {})
        status = v.get("status", "UNKNOWN")
        reason = v.get("reason")
        has_error = status == "UNSUPPORTED"
        nodes[label] = {
            "type": "fact",
            "gloss": data["gloss"],
            "formula": data["formula"],
            "verify_status": status,
            "verify_reason": reason,
            "error": has_error,
        }
        if has_error:
            errors.append({"node": label, "stage": "verify", "message": reason or "unsupported fact"})

    for label, data in parsed["rules"].items():
        v = verify_statuses.get(label, {})
        status = v.get("status", "UNKNOWN")
        reason = v.get("reason")
        has_error = status == "IMPLAUSIBLE"
        nodes[label] = {
            "type": "rule",
            "gloss": data["gloss"],
            "formula": data["formula"],
            "verify_status": status,
            "verify_reason": reason,
            "error": has_error,
        }
        if has_error:
            errors.append({"node": label, "stage": "verify", "message": reason or "implausible rule"})

    inf_labels = sorted(
        parsed["inferences"].keys(),
        key=lambda x: int(re.search(r'\d+', x).group()),
    )
    for label in inf_labels:
        data = parsed["inferences"][label]
        t = tptp_statuses.get(label, {})
        status = t.get("status", "NOT_CHECKED")
        reason = t.get("reason")
        has_error = status == "NOT PROVED" or (
            status.startswith("UNKNOWN") and bool(reason)
        )
        nodes[label] = {
            "type": "inference",
            "gloss": data["gloss"],
            "formula": data["formula"],
            "premises": data["premises"],
            "inference_rule": data["inference_rule"],
            "tptp_status": status,
            "tptp_reason": reason,
            "error": has_error,
        }
        if has_error:
            errors.append({"node": label, "stage": "tptp", "message": reason or "inference not proved"})

    root = inf_labels[-1] if inf_labels else None
    error_stages = sorted({e["stage"] for e in errors})

    return {
        "nodes": nodes,
        "root": root,
        "errors": errors,
        "error_stages": error_stages,
    }
