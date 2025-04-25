# MedHallu Detection

This directory contains code for evaluating LLM performance on the MedHallu medical hallucination detection benchmark.

## Overview

The detection evaluation pipeline assesses various LLMs' ability to distinguish between hallucinated and factual medical information across different difficulty levels.

## Dataset Access

You can access the MedHallu dataset directly from Hugging Face:

```python
from datasets import load_dataset
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled")
```

## Files

- `detection_vllm_notsurecase.py` - Script for evaluating LLMs with a "not sure" option
- `bidirectional_checking.py` - Utility for comparing semantic similarity between answers
- `Mesh.py` - Tool for analyzing MeSH (Medical Subject Headings) categories

## Requirements

- Python 3.8+
- PyTorch
- vLLM
- Transformers
- Pandas
- Scikit-learn
- MeSH XML file (for `Mesh.py`)

## Setup

### File Paths

Replace the following placeholders in `detection_vllm_notsurecase.py`:

```python
df_path = "YOUR_DATASET_PATH"  # Line 304
csv_path = "YOUR_RESULTS_PATH"  # Line 305
```

For `bidirectional_checking.py`:

```python
input_file = "YOUR_INPUT_FILE_PATH"  # Line 143
output_file = "YOUR_OUTPUT_FILE_PATH"  # Line 144
```

For `Mesh.py`:

```python
mesh_xml_file = './desc2025.xml'  # Line 13
```

### Model Configuration

You can customize the list of models to evaluate in `detection_vllm_notsurecase.py`:

```python
models = [
    {'type': 'hf', 'model_name': 'm42-health/Llama3-Med42-8B'},
    # ... other models
]  # Lines 307-320
```

## Running Evaluation

### Main Evaluation

To run the main evaluation script:

```bash
python detection_vllm_notsurecase.py
```

This will:
1. Load the MedHallu dataset
2. Evaluate each model with and without domain knowledge
3. Calculate performance metrics (precision, recall, F1 score)
4. Save results to the specified CSV file

### Bidirectional Checking

To analyze semantic similarity between hallucinated and ground truth answers:

```bash
python bidirectional_checking.py
```

### MeSH Analysis

To analyze MeSH categories in the dataset:

```bash
python Mesh.py
```
Note: You'll need to download the MeSH XML file separately.

## Evaluation Metrics

The evaluation provides:
- Overall precision, recall, and F1 scores
- Performance breakdown by difficulty level (easy, medium, hard)
- Metrics with and without access to medical knowledge
- Percentage of "not sure" responses when available

## Model Support

The detection script supports:
- Hugging Face models via vLLM
- OpenAI models (when configured with API key)
