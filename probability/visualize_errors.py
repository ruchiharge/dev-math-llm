import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fractions import Fraction
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

print(f" Creating visualizations for: {RESULTS_FILE}\n")
with open(RESULTS_FILE, 'r') as f:
    results = json.load(f)

df = pd.DataFrame(results)
def classify_error(row):
    if row['is_correct']:
        return "CORRECT"
    
    expected = row['expected_answer']
    got = row['model_response']
    
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
        
        if diff < 0.001:
            return "ROUNDING_ERROR"
        elif diff < 0.05:
            return "CLOSE_ERROR"
        
        if '/' in expected and '/' not in got:
            return "FORMAT_ERROR"
        elif '/' not in expected and '/' in got:
            return "FORMAT_ERROR"
        
        if '/' in expected and '/' in got:
            exp_frac = Fraction(expected)
            got_frac = Fraction(got)
            if exp_frac == got_frac:
                return "UNSIMPLIFIED"
            else:
                return "WRONG_FRACTION"
        
        return "CALC_ERROR"
        
    except:
        if got == "":
            return "EMPTY"
        return "PARSE_ERROR"

df['error_type'] = df.apply(classify_error, axis=1)

sns.set_style("whitegrid")
sns.set_palette("husl")

fig = plt.figure(figsize=(16, 10))

# 1. Error Type Distribution (Pie Chart)
plt.subplot(2, 3, 1)
error_counts = df['error_type'].value_counts()
colors = ['#2ecc71' if 'CORRECT' in str(x) else '#e74c3c' for x in error_counts.index]
plt.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors)
plt.title('Error Type Distribution', fontsize=14, fontweight='bold')

# 2. Accuracy by Template
plt.subplot(2, 3, 2)
template_acc = df.groupby('template_id')['is_correct'].mean() * 100
bars = plt.bar(template_acc.index, template_acc.values, color='#3498db')
plt.title('Accuracy by Template Type', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=11)
plt.ylim([0, 100])
plt.xticks(rotation=0)
plt.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='50% baseline')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 3. Accuracy by Variation (Fraction vs Decimal)
plt.subplot(2, 3, 3)
var_acc = df.groupby('variation_id')['is_correct'].mean() * 100
var_labels = ['Fraction\n(var_1)', 'Decimal\n(var_2)']
bars = plt.bar(var_labels, var_acc.values, color=['#9b59b6', '#e67e22'])
plt.title('Fraction vs Decimal Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=11)
plt.ylim([0, 100])
plt.axhline(y=50, color='red', linestyle='--', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, var_acc.values)):
    plt.text(bar.get_x() + bar.get_width()/2., val,f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Top 10 Problems with Most Errors
plt.subplot(2, 3, 4)
problem_errors = df[df['is_correct'] == False].groupby('problem_id').size().sort_values(ascending=True).tail(10)
plt.barh(problem_errors.index, problem_errors.values, color='#e74c3c')
plt.title('Top 10 Most Difficult Problems', fontsize=14, fontweight='bold')
plt.xlabel('Number of Errors', fontsize=11)
plt.ylabel('Problem ID', fontsize=11)

# Add value labels
for i, (idx, val) in enumerate(problem_errors.items()):
    plt.text(val + 0.1, i, str(val), va='center', fontsize=9)

# 5. Overall Performance (Bar Chart)
plt.subplot(2, 3, 5)
correct_count = df['is_correct'].sum()
incorrect_count = len(df) - correct_count
bars = plt.bar(['Correct', 'Incorrect'], [correct_count, incorrect_count],color=['#2ecc71', '#e74c3c'], width=0.6)
plt.title('Overall Performance', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=11)

# Add counts and percentages
total = len(df)
for bar, count in zip(bars, [correct_count, incorrect_count]):
    height = bar.get_height()
    pct = (count / total) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height/2,f'{count}\n({pct:.1f}%)', ha='center', va='center',fontsize=12, fontweight='bold', color='white')

# 6. Response Time Distribution
plt.subplot(2, 3, 6)
plt.hist(df['latency_sec'], bins=25, color='#9b59b6', alpha=0.7, edgecolor='black')
plt.axvline(df['latency_sec'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {df["latency_sec"].mean():.2f}s')
plt.axvline(df['latency_sec'].median(), color='green', linestyle='--',
            linewidth=2, label=f'Median: {df["latency_sec"].median():.2f}s')
plt.title('Response Time Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Latency (seconds)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.legend(fontsize=9)
plt.grid(axis='y', alpha=0.3)

# Add overall title
model_name = df['model'].iloc[0]
accuracy = (df['is_correct'].sum() / len(df)) * 100
fig.suptitle(f'Model Performance Analysis: {model_name} (Accuracy: {accuracy:.2f}%)',fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])

output_filename = RESULTS_FILE.replace('_evaluated.json', '_visualization.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f" Visualization saved to: {output_filename}")
plt.show()
print("\n Visualization complete!")