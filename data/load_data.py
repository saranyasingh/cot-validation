import requests
import json
import re
import argparse
from datasets import load_dataset

# ── SVAMP ──────────────────────────────────────────────────────────────────────

url = "https://datasets-server.huggingface.co/rows"
params = {
    "dataset": "ChilleD/SVAMP",
    "config": "default",
    "split": "train",
    "offset": 0,
    "length": 100
}

response = requests.get(url, params=params)
data = response.json()

items = []
for i, entry in enumerate(data["rows"], start=1):
    row = entry["row"]

    # best option: use question_concat if you want full word problem + question together
    question_text = row.get("question_concat")

    # fallback if question_concat is missing
    if not question_text:
        body = row.get("Body", "").strip()
        question = row.get("Question", "").strip()
        question_text = f"{body} {question}".strip()

    items.append({
        "id": f"item_{i:03d}",
        "question": question_text,
        "answer": str(row.get("Answer", "")).strip()
    })

output = {
    "name": "Math Word Problems",
    "items": items
}

with open("math_word_problems.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved to math_word_problems.json")


# ── ProofWriter ────────────────────────────────────────────────────────────────

HF_PROOFWRITER = "D3xter1922/proofwriter-dataset"


def _parse_proofwriter_row(en: str, ro: str, idx: int) -> dict | None:
    """
    Parse one row from D3xter1922/proofwriter-dataset.

    en: "$answer$ ; $proof$ ; $question$ = <q> ; $context$ = sent1: ... sentN: ..."
    ro: "$answer$ = True/False/Unknown ; $proof$ = sentX"

    Sentences starting with "If" are rules; all others are facts.
    Both are stored explicitly so rule_pipeline.py can load them directly
    without relying on the LLM to re-extract them.
    """
    q_match = re.search(r'\$question\$\s*=\s*(.+?)\s*;', en)
    if not q_match:
        return None
    question = q_match.group(1).strip().rstrip('.')

    ctx_match = re.search(r'\$context\$\s*=\s*(.+)$', en, re.DOTALL)
    if not ctx_match:
        return None
    raw_ctx = ctx_match.group(1).strip()
    sentences = [s.strip().rstrip('.') + '.' for s in re.findall(r'sent\d+:\s*(.+?)(?=\s+sent\d+:|$)', raw_ctx)]

    facts = [s for s in sentences if not s.lower().startswith("if")]
    rules = [s for s in sentences if s.lower().startswith("if")]

    ans_match = re.search(r'\$answer\$\s*=\s*(True|False|Unknown)', ro, re.IGNORECASE)
    expected = ans_match.group(1).capitalize() if ans_match else "Unknown"

    return {
        "id":       f"item_{idx:05d}",
        "theory":   " ".join(sentences),   # full theory kept for reference
        "facts":    facts,
        "rules":    rules,
        "question": question,
        "expected": expected,
    }


def load_proofwriter(split: str = "test", limit: int = 50, output: str = "proofwriter.json"):
    """Load ProofWriter from HuggingFace and save to a JSON file."""
    print(f"Loading {HF_PROOFWRITER} ({split} split, up to {limit} items)...")
    ds = load_dataset(HF_PROOFWRITER, split=split)

    items = []
    for i, row in enumerate(ds):
        if len(items) >= limit:
            break
        translation = row.get("translation", {})
        parsed = _parse_proofwriter_row(
            translation.get("en", ""),
            translation.get("ro", ""),
            i + 1,
        )
        if parsed:
            items.append(parsed)

    out = {"name": "ProofWriter", "items": items}
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(items)} items to {output}")


def load_gsm8k(split: str = "test", limit: int = None, output: str = "dataset.json"):
    """Load GSM8K from HuggingFace and save to a JSON file."""
    print(f"Loading openai/gsm8k ({split} split)...")
    ds = load_dataset("openai/gsm8k", "main", split=split)

    items = []
    for i, row in enumerate(ds):
        if limit is not None and len(items) >= limit:
            break
        raw_answer = row.get("answer", "")
        # Extract final numeric answer after ####
        match = re.search(r'####\s*(.+)', raw_answer)
        final_answer = match.group(1).strip() if match else raw_answer.strip()
        items.append({
            "id": f"item_{i+1:04d}",
            "question": row["question"].strip(),
            "answer": final_answer,
        })

    out = {"name": "GSM8K", "items": items}
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(items)} items to {output}")


if __name__ == "__main__":
    import sys
    if "--proofwriter" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--proofwriter", action="store_true")
        parser.add_argument("--split", default="test")
        parser.add_argument("--limit", "-n", type=int, default=50)
        parser.add_argument("--output", "-o", default="proofwriter.json")
        args = parser.parse_args()
        load_proofwriter(split=args.split, limit=args.limit, output=args.output)
    elif "--gsm8k" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--gsm8k", action="store_true")
        parser.add_argument("--split", default="test")
        parser.add_argument("--limit", "-n", type=int, default=None)
        parser.add_argument("--output", "-o", default="dataset.json")
        args = parser.parse_args()
        load_gsm8k(split=args.split, limit=args.limit, output=args.output)
