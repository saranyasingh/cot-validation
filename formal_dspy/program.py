"""
The formal half of the main pipeline as a DSPy program, plus the metric GEPA optimizes.

Only the two prompts that feed the provers are optimizable: FOL extraction
(STRUCTURED_FOL_TEMPLATE) and FOL-to-TPTP conversion (TPTP_TEMPLATE). The arithmetic
check (Vampire) and the inference check (E/Vampire) are the existing deterministic code.
The fact/rule LLM judge is not part of this program.
"""

import re
from fractions import Fraction

import dspy

from formal_dspy import lm as _lm  # noqa: F401  (puts main_pipeline/ on sys.path)
from main_pipeline.arithmetic import check_operations
from main_pipeline.main import STRUCTURED_FOL_TEMPLATE
from main_pipeline.tptp import TPTP_TEMPLATE, check_tptp, normalize_numbers

FOL_INSTRUCTIONS = STRUCTURED_FOL_TEMPLATE.split("=== CHAIN OF THOUGHT REASONING TO CONVERT ===")[0].strip()
TPTP_INSTRUCTIONS = (TPTP_TEMPLATE.replace("{structured_fol}", "(given as the structured_fol input)")
                     .replace("=== CONVERT NOW ===", "").strip())


class ExtractFOL(dspy.Signature):
    cot_text: str = dspy.InputField(desc="the chain-of-thought reasoning to convert")
    structured_fol: str = dspy.OutputField(
        desc="the ## VOCABULARY, ## FACTS, ## OPERATIONS, ## RULES and ## INFERENCES sections, "
             "one 'LABEL: gloss :: formula' line per item")


class ConvertTPTP(dspy.Signature):
    structured_fol: str = dspy.InputField(desc="structured first-order logic to convert")
    tptp: str = dspy.OutputField(desc="only TPTP comments (% ...) and fof(...) statements")


def _strip_fences(text: str) -> str:
    return re.sub(r"^```\w*\s*\n|\n?```\s*$", "", (text or "").strip())


class FormalVerifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_fol = dspy.Predict(ExtractFOL.with_instructions(FOL_INSTRUCTIONS))
        self.to_tptp = dspy.Predict(ConvertTPTP.with_instructions(TPTP_INSTRUCTIONS))

    def extract(self, cot_text: str) -> str:
        return _strip_fences(self.extract_fol(cot_text=cot_text).structured_fol)

    def convert(self, structured_fol: str) -> str:
        return normalize_numbers(_strip_fences(self.to_tptp(structured_fol=structured_fol).tptp))

    def forward(self, cot_text: str):
        fol = self.extract(cot_text)
        arith_errors, arith_report, _ = check_operations(fol)
        tptp, tptp_result = "", None
        if not arith_errors:
            tptp = self.convert(fol)
            tptp_result = check_tptp(tptp)
        verdict, problems = formal_verdict(arith_report, tptp_result)
        return dspy.Prediction(structured_fol=fol, arithmetic_report=arith_report, tptp=tptp,
                               tptp_report=tptp_result["report"] if tptp_result else "",
                               verdict=verdict, problems=problems)


# ─── Verdict ──────────────────────────────────────────────────────────────────

def formal_verdict(arith_report: str, tptp_result: dict | None) -> tuple[str, list[str]]:
    """
    proved:  every OP is correct and every inference is proved
    refuted: a prover showed an OP is wrong or an inference does not follow
    broken:  the formalization could not be checked (malformed OP, TPTP syntax error,
             no inferences, timeout, unknown, prover error)
    """
    arith = dict(re.findall(r"^(OP-\d+): (CORRECT|INCORRECT|MALFORMED|UNKNOWN)", arith_report, re.M))
    broken = [f"{op} {s}" for op, s in arith.items() if s in ("MALFORMED", "UNKNOWN")]
    if broken:
        return "broken", broken
    wrong_ops = [op for op, s in arith.items() if s == "INCORRECT"]
    if wrong_ops:
        return "refuted", [f"{op} INCORRECT" for op in wrong_ops]
    if tptp_result is None:
        return "broken", ["TPTP was not checked"]
    if tptp_result.get("syntax_error"):
        return "broken", ["TPTP syntax error"]
    statuses = tptp_result.get("statuses") or {}
    if not statuses:
        return "broken", ["no inferences were checked"]
    broken = [f"{inf} {s}" for inf, s in statuses.items() if s not in ("PROVED", "NOT PROVED")]
    if broken:
        return "broken", broken
    not_proved = [inf for inf, s in statuses.items() if s == "NOT PROVED"]
    if not_proved:
        return "refuted", [f"{inf} NOT PROVED" for inf in not_proved]
    return "proved", []


# ─── Faithfulness ─────────────────────────────────────────────────────────────

WORD_NUMBERS = {"half": "0.5", "quarter": "0.25", "twice": "2", "double": "2", "triple": "3",
                "dozen": "12", "hundred": "100", "thousand": "1000", "one": "1", "two": "2",
                "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
                "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
FORMULA_NUMBER_RE = re.compile(r"(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?(?![A-Za-z_\d])")


def _numbers(text: str) -> set[Fraction]:
    text = text.replace(",", "")
    found = {Fraction(n) for n in NUMBER_RE.findall(text)}
    fractions = re.findall(r"(\d+)\s*/\s*(\d+)", text) + re.findall(r"\\[dt]?frac\{(\d+)\}\{(\d+)\}", text)
    found |= {Fraction(int(a), int(b)) for a, b in fractions if int(b)}
    return found


def _matches(token: str, allowed: set[Fraction]) -> bool:
    """Exact match, or a rounded restatement (2/3 written as 0.6667)."""
    value = Fraction(token)
    if value in allowed:
        return True
    decimals = len(token.split(".")[1]) if "." in token else 0
    tolerance = Fraction(1, 2 * 10 ** decimals)
    return decimals > 0 and any(abs(value - x) <= tolerance for x in allowed)


def unfaithful_numbers(structured_fol: str, question: str, cot_text: str) -> list[str]:
    """Numbers in FACT/OP formulas that appear nowhere in the question or CoT (also as x/100)."""
    source = f"{question}\n{cot_text}"
    allowed = _numbers(source) | {Fraction(0), Fraction(1)}
    allowed |= {Fraction(v) for w, v in WORD_NUMBERS.items() if re.search(rf"\b{w}\b", source, re.I)}
    allowed |= {x / 100 for x in list(allowed)}
    bad = []
    for formula in re.findall(r"^(?:FACT|OP)-\d+:.*?::\s*(.+)$", structured_fol, re.M):
        for n in FORMULA_NUMBER_RE.findall(formula):
            if not _matches(n, allowed) and n not in bad:
                bad.append(n)
    return bad


# ─── Metric ───────────────────────────────────────────────────────────────────

INCORRECT_BUT_PROVED_SCORE = 0.5


def score_prediction(gold, pred) -> tuple[float, str]:
    """
    Correct CoT:   1 if proved, else 0.
    Incorrect CoT: 1 if refuted, 0.5 if proved (some errors, like misreading the problem,
                   are not formal errors), 0 if broken.
    Any FACT/OP number not found in the question or CoT scores 0, as does a failed run.
    """
    if getattr(pred, "verdict", None) is None:
        return 0.0, "failed"
    bad = unfaithful_numbers(pred.structured_fol, gold.question, gold.cot_text)
    if bad:
        return 0.0, f"unfaithful: {', '.join(bad)}"
    if gold.cot_correct:
        return (1.0 if pred.verdict == "proved" else 0.0), pred.verdict
    return {"refuted": 1.0, "proved": INCORRECT_BUT_PROVED_SCORE, "broken": 0.0}[pred.verdict], pred.verdict


def metric(gold, pred, trace=None) -> float:
    return score_prediction(gold, pred)[0]


def _clip(text: str, n: int = 1500) -> str:
    return text if len(text) <= n else text[:n] + "\n... [truncated]"


def feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score, outcome = score_prediction(gold, pred)
    if outcome == "failed":
        return dspy.Prediction(score=0.0, feedback="The program failed before producing a checkable "
                               "formalization: the model's output could not be parsed into the output field.")
    truth = ("The CoT's final answer is CORRECT, so a faithful formalization should prove every "
             "operation and inference." if gold.cot_correct else
             "The CoT's final answer is WRONG. A faithful formalization exposes an arithmetic slip "
             "(an INCORRECT operation) or a step that does not follow (an inference NOT PROVED). If the "
             "error is a misreading of the problem, the formal checks cannot catch it and that is fine.")
    lines = [truth, f"Outcome: {pred.verdict}" + (f" ({'; '.join(pred.problems)})" if pred.problems else "")]
    if outcome.startswith("unfaithful"):
        lines.append(f"These numbers in FACT/OP formulas are not in the problem or the CoT: "
                     f"{outcome.split(': ', 1)[1]}. Copy numbers exactly; never correct or invent them.")
    lines.append("Never change numbers, drop steps or add premises to force an outcome; the formalization "
                 "must mirror the CoT step by step.")
    if pred_name in (None, "extract_fol"):
        lines.append("Arithmetic check:\n" + _clip(pred.arithmetic_report))
    if pred_name in (None, "to_tptp") and pred.tptp_report:
        lines.append("Inference check (each inf_N proved from the axioms and earlier inferences):\n"
                     + _clip(pred.tptp_report))
    return dspy.Prediction(score=score, feedback="\n\n".join(lines))
