# MedHallu Dataset Generation

This directory contains the code for generating the MedHallu dataset, which consists of hallucinated medical question-answer pairs derived from PubMedQA.

## Overview

The dataset generation pipeline follows these steps:

1. Extract question-answer pairs from PubMedQA
2. Generate hallucinated answers using LLMs
3. Filter and classify hallucinations based on difficulty
4. Apply TextGrad optimizations to improve failed generations
5. Categorize hallucinations and save the final dataset

## Files

- `generation.py` - Main script for dataset generation
- `Prompts/system_prompt.txt` - Prompt template for hallucination generation
- `Prompts/system_prompt_medical.txt` - Medical-specific prompt template
- `Prompts/system_prompt_detection.txt` - Prompt for hallucination detection

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Hugging Face Datasets
- vLLM
- Sentence Transformers
- TextGrad

## Setup

### API Keys

Replace the placeholder for OpenAI API key in `generation.py`:

```python
openai_client = OpenAI(api_key="YOUR_OPENAI_API_KEY")  # Line 397
```

### File Paths

Replace the following placeholders in `generation.py`:

```python
OUTPUT_FILE = os.path.join("YOUR_OUTPUT_PATH", "medhallu_dataset.csv")  # Line 127
CHECKPOINT_FILE = os.path.join("YOUR_CHECKPOINT_PATH", "medhallu_checkpoint.csv")  # Line 128
```

### Model Configuration

You can modify the model configurations if needed:

```python
GENERATOR_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  # Line 121
DISCRIMINATOR_CONFIGS = [...]  # Lines 139-158
```

## Running the Generation Pipeline

To generate the dataset, simply run:

```bash
python generation.py
```

This will:
1. Load the PubMedQA dataset
2. Initialize LLM models for generation and discrimination
3. Generate hallucinated answers for each question
4. Apply filtering and optimization steps
5. Save the resulting dataset to the specified output file

## Parameters

Key parameters that can be modified:

- `NUM_GENERATIONS` (Line 120): Number of generation attempts per question
- `BATCH_SIZE` (Line 122): Number of questions to process
- `GENERATOR_TEMPERATURE` (Line 123): Temperature for hallucination generation
- `DISCRIMINATOR_TEMPERATURE` (Line 124): Temperature for discriminator models
- `TOP_P` (Line 125): Top-p sampling parameter
- `MAX_NEW_TOKENS` (Line 126): Maximum tokens for generated text

## Output Format

The generated dataset will include the following for each entry:

- Original question and ground truth answer
- Generated hallucinated answers
- Hallucination justifications
- Discriminator results
- Difficulty level (easy, medium, hard)
- Least similar answer (for fallback cases)

## Checkpoint System

The generation process includes a checkpoint system that saves progress periodically, allowing you to resume generation if interrupted.
