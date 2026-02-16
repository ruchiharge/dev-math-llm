# LLM Evaluation Framework for Mathematical Reasoning  
## Probability Dataset (Ongoing Multi-Dataset Benchmarking Project)

---

## Overview

This project implements a **systematic evaluation pipeline for Large Language Models (LLMs)** on structured mathematical reasoning datasets.

Currently implemented and evaluated on:

- Probability dataset  
- Models tested:
  - `gemma2:2b`
  - `gemma3:4b`

The framework is designed to be **dataset-agnostic and extensible**, enabling evaluation on additional mathematical domains such as:

- Derivatives
- Integrals
- Algebra
- Limits
- Combinatorics
- And other structured math datasets (10+ planned)

This is an ongoing benchmarking project focused on:

- Accuracy evaluation
- Error classification
- Format analysis (fraction vs decimal)
- Latency analysis
- Template-based performance breakdown
- Error pattern mining

---

## Project Goals

- Evaluate LLM mathematical reasoning reliability
- Detect structured reasoning weaknesses
- Classify types of numerical errors
- Compare model variants (2B vs 4B)
- Build reusable evaluation + visualization pipeline
- Extend to multiple mathematical domains

---

## Pipeline Architecture

The framework consists of four main stages:

```
1. Model Inference
2. Answer Normalization
3. Evaluation & Accuracy Computation
4. Error Classification & Visualization
```

---

## Project Structure

```
/llm-math-evaluation
│
├── probability_test.csv
├── derivatives_test.csv        (planned)
├── integrals_test.csv          (planned)
├── ...
│
├── run_single_model.py
├── evaluate_results.py
├── visualize_errors.py
├── analyze_errors.py
│
├── results_gemma2_2b.json
├── results_gemma3_4b.json
├── results_*_evaluated.json
├── evaluation_report.json
└── README.md
```

---

## Dataset Format

Each CSV file must contain:

| Column Name       | Description |
|-------------------|-------------|
| problem_id        | Unique identifier |
| problem_type      | Category |
| template_id       | Template group |
| variation_id      | Fraction / Decimal variation |
| input             | Problem text |
| expected_answer   | Ground truth |

---

## Stage 1: Model Inference

File: `run_single_model.py`

- Sends structured prompts to local Ollama server
- Uses deterministic decoding (`temperature = 0.0`)
- Extracts only numerical answers
- Cleans output (fractions, decimals, integers)
- Measures latency
- Saves raw responses to JSON

Example execution:

```bash
python run_single_model.py
```

Output:

```
results_gemma3_4b.json
```

---

## Stage 2: Evaluation

File: `evaluate_results.py`

Performs:

- Answer normalization
- Fraction ↔ Decimal equivalence checking
- Tolerance-based comparison (±0.005)
- Correctness flagging
- Template-wise accuracy
- Variation-wise accuracy
- Latency analysis

Outputs:

```
evaluation_report.json
results_gemma3_4b_evaluated.json
```

Run:

```bash
python evaluate_results.py
```

---

## Stage 3: Error Analysis

File: `analyze_errors.py`

Classifies errors into:

- CORRECT
- ROUNDING_ERROR
- CLOSE_ERROR
- FORMAT_ERROR_FRACTION_TO_DECIMAL
- FORMAT_ERROR_DECIMAL_TO_FRACTION
- EQUIVALENT_FRACTION
- WRONG_FRACTION
- CALCULATION_ERROR
- EMPTY_RESPONSE
- PARSING_ERROR

Also generates:

- Problem-specific error breakdown
- Template-wise performance
- Variation-wise performance
- Error matrix
- CSV export of detailed analysis

Run:

```bash
python analyze_errors.py results_gemma3_4b_evaluated.json
```

---

## Stage 4: Visualization

File: `visualize_errors.py`

Generates:

- Error type distribution (Pie chart)
- Accuracy by template
- Fraction vs Decimal comparison
- Most difficult problems
- Overall correct vs incorrect distribution
- Response time histogram

Output:

```
*_visualization.png
```

Run:

```bash
python visualize_errors.py results_gemma3_4b_evaluated.json
```

---

## Current Results (Probability Dataset)

Models tested:

- Gemma 2B
- Gemma 4B

Metrics tracked:

- Accuracy
- Error rate
- Template breakdown
- Variation breakdown
- Average latency
- Error type distribution

---

## Key Features

- Deterministic evaluation
- Numerical normalization
- Fraction simplification detection
- Rounding tolerance analysis
- Format sensitivity detection
- Error magnitude computation
- Latency tracking
- Reusable across datasets
- Fully local inference via Ollama

---

## Extending to Other Datasets (Planned)

To evaluate additional datasets (e.g., derivatives, integrals):

1. Replace CSV file in `run_single_model.py`
2. Update prompt template if necessary
3. Run full pipeline:
   - Inference
   - Evaluation
   - Error analysis
   - Visualization

No architectural changes required.

This design allows benchmarking across multiple mathematical domains with minimal modification.

---

## Requirements

- Python 3.9+
- Ollama installed and running
- Local models available:
  - gemma2:2b
  - gemma3:4b

Python libraries:

```bash
pip install pandas matplotlib seaborn requests
```

---

## Research Direction (Ongoing Work)

Planned extensions:

- Compare small vs medium vs large LLMs
- Cross-domain generalization study
- Error-type transfer analysis
- Prompt engineering experiments
- Few-shot vs zero-shot comparison
- Chain-of-thought vs direct answer comparison
- Statistical significance testing
- Automated benchmarking across 10+ math domains

---

## Current Status

- Probability dataset fully evaluated
- Error taxonomy implemented
- Visualization pipeline complete
- Multi-dataset expansion in progress

---

## Summary

This project builds a **robust evaluation framework for mathematical reasoning in LLMs**, starting with probability and extending to a broader mathematical benchmark suite.

It focuses not just on accuracy, but on:

- Why the model fails
- How it fails
- What type of reasoning errors occur
- Format sensitivity
- Numerical precision issues
- Structural weaknesses

This is an evolving research and benchmarking framework.
