import json
import pandas as pd
from fractions import Fraction

def normalize_answer(answer: str) -> str:
    """Normalize answers for comparison"""
    answer = answer.strip().lower()
    
    # Try to parse as fraction
    if '/' in answer:
        try:
            frac = Fraction(answer)
            return f"{frac.numerator}/{frac.denominator}"
        except:
            pass
    
    # Try to parse as decimal
    try:
        decimal_val = float(answer)
        return f"{decimal_val:.3f}"
    except:
        pass
    
    return answer

def answers_match(model_answer: str, expected_answer: str) -> bool:
    """Check if model answer matches expected answer"""
    model_norm = normalize_answer(model_answer)
    expected_norm = normalize_answer(expected_answer)
    
    # Direct match
    if model_norm == expected_norm:
        return True
    
    # Try fraction to decimal comparison
    try:
        if '/' in expected_answer:
            expected_decimal = float(Fraction(expected_answer))
        else:
            expected_decimal = float(expected_answer)
        
        if '/' in model_answer:
            model_decimal = float(Fraction(model_answer))
        else:
            model_decimal = float(model_answer)
        
        # Check if they're close (within 0.005)
        return abs(model_decimal - expected_decimal) < 0.005
    except:
        return False

# Change the filename here
with open("results_gemma3_4b.json", "r", encoding="utf-8") as f:
    results = json.load(f)

total = len(results)
correct = 0
errors = 0
incorrect_details = []

for result in results:
    model_ans = result.get("model_response", "")
    expected_ans = result.get("expected_answer", "")
    
    if "ERROR" in result.get("model_response_raw", ""):
        errors += 1
        continue
    
    if answers_match(model_ans, expected_ans):
        correct += 1
        result["is_correct"] = True
    else:
        result["is_correct"] = False
        incorrect_details.append({
            "problem_id": result["problem_id"],
            "input": result["input"],
            "expected": expected_ans,
            "got": model_ans
        })

accuracy = (correct / total) * 100 if total > 0 else 0
error_rate = (errors / total) * 100 if total > 0 else 0

avg_latency = sum(r["latency_sec"] for r in results) / total if total > 0 else 0

print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Model: {results[0]['model']}")
print(f"Total Questions: {total}")
print(f"Correct Answers: {correct}")
print(f"Incorrect Answers: {total - correct - errors}")
print(f"Errors: {errors}")
print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Error Rate: {error_rate:.2f}%")
print(f"Average Latency: {avg_latency:.3f} seconds")
print("=" * 60)

if incorrect_details:
    print("\n SAMPLE INCORRECT ANSWERS (first 10):")
    print("-" * 60)
    for detail in incorrect_details[:10]:
        print(f"\nProblem: {detail['problem_id']}")
        print(f"Question: {detail['input']}")
        print(f"Expected: {detail['expected']}")
        print(f"Got: {detail['got']}")

print("\n\n" + "=" * 60)
print("BREAKDOWN BY TEMPLATE")
print("=" * 60)

df_results = pd.DataFrame(results)
template_stats = df_results.groupby('template_id').agg({
    'is_correct': ['sum', 'count']
}).reset_index()

template_stats.columns = ['template_id', 'correct', 'total']
template_stats['accuracy'] = (template_stats['correct'] / template_stats['total'] * 100).round(2)

print(template_stats.to_string(index=False))

print("\n" + "=" * 60)
print("BREAKDOWN BY VARIATION")
print("=" * 60)

variation_stats = df_results.groupby('variation_id').agg({
    'is_correct': ['sum', 'count']
}).reset_index()

variation_stats.columns = ['variation_id', 'correct', 'total']
variation_stats['accuracy'] = (variation_stats['correct'] / variation_stats['total'] * 100).round(2)

print(variation_stats.to_string(index=False))

output_file = "evaluation_report.json"
evaluation_report = {
    "model": results[0]['model'],
    "total_questions": total,
    "correct": correct,
    "incorrect": total - correct - errors,
    "errors": errors,
    "accuracy_percent": round(accuracy, 2),
    "error_rate_percent": round(error_rate, 2),
    "avg_latency_sec": round(avg_latency, 3),
    "template_breakdown": template_stats.to_dict('records'),
    "variation_breakdown": variation_stats.to_dict('records'),
    "incorrect_samples": incorrect_details
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(evaluation_report, f, indent=2)

print(f"\n Detailed evaluation report saved to: {output_file}")
# Change the filename here
with open("results_gemma3_4b_evaluated.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f" Updated results with correctness flags saved to: results_gemma3_4b_evaluated.json")
# Change the filename here