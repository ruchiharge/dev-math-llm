import pandas as pd
import requests
import json
import time
from datetime import datetime, timezone
import re

MODEL_NAME = "gemma3:4b"
CSV_PATH = "probability_test.csv"
OUTPUT_JSON = "results_gemma3_4b.json" #here need to change the filename whenever
OLLAMA_URL = "http://localhost:11434/api/generate"

def extract_answer(text: str) -> str:
    if not text:
        return ""
    
    # Remove common phrases and clean up
    text = text.lower()
    text = text.replace("the probability is", "")
    text = text.replace("the answer is", "")
    text = text.replace("answer:", "")
    text = text.replace("final answer:", "")
    text = text.strip()
    text = text.strip('.')
    text = text.strip(',')

    # Look for fraction pattern
    frac = re.search(r"\b\d+/\d+\b", text)
    if frac:
        return frac.group(0)

    # Look for decimal pattern
    dec = re.search(r"\b0\.\d+\b", text)
    if dec:
        return dec.group(0)

    # Look for any number
    num = re.search(r"\b\d+\b", text)
    if num:
        return num.group(0)

    return text.strip()

df = pd.read_csv(CSV_PATH)

results = []

for idx, row in df.iterrows():
    question = row["input"]
    
    prompt = f"""Solve this probability problem. Give ONLY the numerical answer.

Examples:
Q: What is the probability of rolling a 3 on a fair die?
A: 1/6

Q: What is the probability of flipping heads? (as decimal)
A: 0.5

Now solve:
Q: {question}
A:"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "options": {
            "temperature": 0.0,
            "num_predict": 20
        },
        "stream": False
    }

    start_time = time.time()

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        output = response.json()

        raw_answer = output.get("response", "").strip()
        clean_answer = extract_answer(raw_answer)

    except Exception as e:
        raw_answer = f"ERROR: {str(e)}"
        clean_answer = ""

    latency = round(time.time() - start_time, 3)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "problem_id": row["problem_id"],
        "problem_type": row["problem_type"],
        "template_id": row["template_id"],
        "variation_id": row["variation_id"],
        "input": row["input"],
        "expected_answer": row["expected_answer"],
        "model_response_raw": raw_answer,
        "model_response": clean_answer,
        "latency_sec": latency
    }

    results.append(result)

    is_correct = "✓" if clean_answer == row["expected_answer"] else "✗"
    print(f"[{idx+1}/{len(df)}] {is_correct} | Got: {clean_answer} | Expected: {row['expected_answer']} | Time: {latency}s")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

correct = sum(1 for r in results if r["model_response"] == r["expected_answer"])
accuracy = (correct / len(results)) * 100

print("\n" + "="*60)
print(" Evaluation complete")
print(f" Results saved to: {OUTPUT_JSON}")
print(f" Quick Accuracy: {correct}/{len(results)} ({accuracy:.1f}%)")
print("="*60)