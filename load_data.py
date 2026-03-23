import requests
import json

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