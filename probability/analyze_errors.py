import json
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
import os
if len(sys.argv) > 1:
    RESULTS_FILE = sys.argv[1]
else:
    json_files = [f for f in os.listdir('.') if f.endswith('_evaluated.json')]
    if not json_files:
        print(" No evaluated results file found!")
        sys.exit(1)
    RESULTS_FILE = json_files[0]
    print(f" Analyzing: {RESULTS_FILE}\n")
with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# ==============================
# ERROR CLASSIFICATION
# ==============================
def classify_error(row):
    """Classify the type of error"""
    if row['is_correct']:
        return "CORRECT"
    
    expected = row['expected_answer']
    got = row['model_response']
    
    # Extract numbers for analysis
    try:
        if '/' in expected:
            expected_val = float(Fraction(expected))
        else:
            expected_val = float(expected)
        
        if '/' in got:
            got_val = float(Fraction(got))
        else:
            got_val = float(got)
        
        diff = abs(expected_val - got_val)
        
        if diff < 0.01:
            return "ROUNDING_ERROR"  # just rounding
        elif diff < 0.05:
            return "CLOSE_ERROR"  # Close but not quite
        
        #(fraction vs decimal)
        if '/' in expected and '/' not in got:
            return "FORMAT_ERROR_FRACTION_TO_DECIMAL"
        elif '/' not in expected and '/' in got:
            return "FORMAT_ERROR_DECIMAL_TO_FRACTION"
        
        # Check if answer is mathematically equivalent but not simplified
        if '/' in expected and '/' in got:
            exp_frac = Fraction(expected)
            got_frac = Fraction(got)
            if exp_frac == got_frac:
                return "EQUIVALENT_FRACTION"
            else:
                return "WRONG_FRACTION"
        
        return "CALCULATION_ERROR"  # Wrong calculation
        
    except:
        # If conversion fails
        if got == "":
            return "EMPTY_RESPONSE"
        return "PARSING_ERROR"

# Apply classification
df['error_type'] = df.apply(classify_error, axis=1)

# ==============================
# ANALYSIS 1: ERROR TYPE DISTRIBUTION
# ==============================
print("=" * 70)
print(" ERROR TYPE DISTRIBUTION")
print("=" * 70)

error_counts = df['error_type'].value_counts()
error_percentages = (df['error_type'].value_counts(normalize=True) * 100).round(2)

error_df = pd.DataFrame({
    'Count': error_counts,
    'Percentage': error_percentages
})

print(error_df.to_string())
print()

# ==============================
# ANALYSIS 2: PROBLEM-SPECIFIC ERRORS
# ==============================
print("=" * 70)
print(" ERRORS BY PROBLEM TYPE")
print("=" * 70)

incorrect_df = df[df['is_correct'] == False]
problem_errors = incorrect_df.groupby('problem_id').agg({
    'is_correct': 'count',
    'error_type': lambda x: x.mode()[0] if len(x) > 0 else None
}).rename(columns={'is_correct': 'error_count', 'error_type': 'most_common_error'})

problem_errors = problem_errors.sort_values('error_count', ascending=False)

print(problem_errors.head(10).to_string())
print()

# ==============================
# ANALYSIS 3: ROUNDING & CLOSE ERRORS
# ==============================
print("=" * 70)
print(" ROUNDING & CLOSE ERRORS ANALYSIS")
print("=" * 70)

rounding_errors = df[df['error_type'].isin(['ROUNDING_ERROR', 'CLOSE_ERROR'])]
if len(rounding_errors) > 0:
    print(f"Total Rounding/Close Errors: {len(rounding_errors)}")
    print("\nSamples:")
    for _, row in rounding_errors.head(5).iterrows():
        print(f"  Problem: {row['problem_id']}")
        print(f"  Expected: {row['expected_answer']}")
        print(f"  Got: {row['model_response']}")
        print(f"  Type: {row['error_type']}\n")
else:
    print(" No rounding or close errors!")
print()

# ==============================
# ANALYSIS 4: FORMAT ERRORS
# ==============================
print("=" * 70)
print(" FORMAT ERRORS (Fraction ↔ Decimal)")
print("=" * 70)

format_errors = df[df['error_type'].str.contains('FORMAT_ERROR', na=False)]
if len(format_errors) > 0:
    print(f"Total Format Errors: {len(format_errors)}")
    print(f"  - Fraction→Decimal: {len(df[df['error_type'] == 'FORMAT_ERROR_FRACTION_TO_DECIMAL'])}")
    print(f"  - Decimal→Fraction: {len(df[df['error_type'] == 'FORMAT_ERROR_DECIMAL_TO_FRACTION'])}")
    print("\nSamples:")
    for _, row in format_errors.head(5).iterrows():
        print(f"  Expected: {row['expected_answer']} | Got: {row['model_response']}")
else:
    print(" No format errors!")
print()

# ==============================
# ANALYSIS 5: EQUIVALENT FRACTIONS
# ==============================
print("=" * 70)
print(" EQUIVALENT FRACTION ERRORS (Not Simplified)")
print("=" * 70)

equiv_errors = df[df['error_type'] == 'EQUIVALENT_FRACTION']
if len(equiv_errors) > 0:
    print(f"Total: {len(equiv_errors)}")
    print("\nExamples of unsimplified fractions:")
    for _, row in equiv_errors.head(10).iterrows():
        print(f"  Expected: {row['expected_answer']} | Got: {row['model_response']} | Problem: {row['problem_id']}")
else:
    print(" No equivalent fraction errors!")
print()

# ==============================
# ANALYSIS 6: CALCULATION ERRORS
# ==============================
print("=" * 70)
print(" ACTUAL CALCULATION ERRORS")
print("=" * 70)

calc_errors = df[df['error_type'] == 'CALCULATION_ERROR']
if len(calc_errors) > 0:
    print(f"Total Calculation Errors: {len(calc_errors)}")
    print("\nMost problematic questions:")
    
    for _, row in calc_errors.head(10).iterrows():
        print(f"\n  Problem: {row['problem_id']}")
        print(f"  Question: {row['input'][:80]}...")
        print(f"  Expected: {row['expected_answer']}")
        print(f"  Got: {row['model_response']}")
        try:
            if '/' in row['expected_answer']:
                exp_val = float(Fraction(row['expected_answer']))
            else:
                exp_val = float(row['expected_answer'])
            
            if '/' in row['model_response']:
                got_val = float(Fraction(row['model_response']))
            else:
                got_val = float(row['model_response'])
            
            error_magnitude = abs(exp_val - got_val)
            print(f"  Error Magnitude: {error_magnitude:.4f}")
        except:
            print(f"  Error Magnitude: Could not calculate")
else:
    print(" No calculation errors!")
print()
print("=" * 70)
print(" ACCURACY BY TEMPLATE & VARIATION")
print("=" * 70)

template_accuracy = df.groupby('template_id')['is_correct'].agg(['sum', 'count', 'mean'])
template_accuracy['accuracy_%'] = (template_accuracy['mean'] * 100).round(2)
template_accuracy = template_accuracy[['sum', 'count', 'accuracy_%']]
template_accuracy.columns = ['Correct', 'Total', 'Accuracy (%)']

print("By Template:")
print(template_accuracy.to_string())
print()

variation_accuracy = df.groupby('variation_id')['is_correct'].agg(['sum', 'count', 'mean'])
variation_accuracy['accuracy_%'] = (variation_accuracy['mean'] * 100).round(2)
variation_accuracy = variation_accuracy[['sum', 'count', 'accuracy_%']]
variation_accuracy.columns = ['Correct', 'Total', 'Accuracy (%)']

print("By Variation:")
print(variation_accuracy.to_string())
print()
print("=" * 70)
print("  ERROR PATTERN MATRIX")
print("=" * 70)

error_matrix = pd.crosstab(df['problem_id'], df['error_type'], margins=True)
print(error_matrix.to_string())
print()
output_file = RESULTS_FILE.replace('_evaluated.json', '_error_analysis.csv')
analysis_df = df[['problem_id', 'template_id', 'variation_id', 'input','expected_answer', 'model_response', 'is_correct', 'error_type']]
analysis_df.to_csv(output_file, index=False)
print("=" * 70)
print(f" Detailed analysis saved to: {output_file}")
print("=" * 70)
print("\n" + "=" * 70)
print(" SUMMARY STATISTICS")
print("=" * 70)
total = len(df)
correct = df['is_correct'].sum()
incorrect = total - correct
print(f"Model: {df['model'].iloc[0]}")
print(f"Total Questions: {total}")
print(f"Correct: {correct} ({correct/total*100:.2f}%)")
print(f"Incorrect: {incorrect} ({incorrect/total*100:.2f}%)")
print()
print("Error Breakdown:")
print(f"  - Rounding Errors: {len(df[df['error_type'] == 'ROUNDING_ERROR'])}")
print(f"  - Close Errors: {len(df[df['error_type'] == 'CLOSE_ERROR'])}")
print(f"  - Format Errors: {len(format_errors)}")
print(f"  - Equivalent Fractions: {len(equiv_errors)}")
print(f"  - Calculation Errors: {len(calc_errors)}")
print(f"  - Other Errors: {len(df[~df['error_type'].isin(['CORRECT', 'ROUNDING_ERROR', 'CLOSE_ERROR', 'FORMAT_ERROR_FRACTION_TO_DECIMAL', 'FORMAT_ERROR_DECIMAL_TO_FRACTION', 'EQUIVALENT_FRACTION', 'CALCULATION_ERROR'])])}")
print("=" * 70)