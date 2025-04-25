# MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MedHallu is the first benchmark specifically designed for medical hallucination detection in Large Language Models (LLMs). This repository contains the dataset and code for detecting hallucinations in medical question-answering systems.

## Key Features

- **10,000 high-quality QA pairs** derived from PubMedQA, with systematically generated hallucinated answers
- **Multi-level difficulty classification** (easy, medium, hard) based on hallucination subtlety
- **Four medical-specific hallucination categories**:
  - Misinterpretation of Question
  - Incomplete Information
  - Mechanism and Pathway Misattribution
  - Methodological and Evidence Fabrication
- **Comprehensive evaluation tools** for benchmarking LLM performance

## Dataset and Code

The repository includes:
- Dataset generation pipeline
- Detection evaluation scripts
- Bidirectional entailment checking tools
- Medical category (MeSH) analysis utilities

## Key Findings

- State-of-the-art LLMs (including GPT-4o, Llama-3.1, UltraMedical) struggle with hard hallucination categories (best F1 score: 0.625)
- General-purpose LLMs outperform medical fine-tuned LLMs when provided with domain knowledge
- Harder-to-detect hallucinations are semantically closer to ground truth
- Adding a "not sure" response option improves precision by up to 38%

## Usage

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- vLLM
- Sentence-Transformers

### Quick Start

Clone the repository:
```bash
git clone https://github.com/medhallu/medhallu.git
cd medhallu
```

## Citation

```
@article{pandit2025medhallu,
  title={MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models},
  author={Pandit, Shrey and Xu, Jiawei and Hong, Junyuan and Wang, Zhangyang and Chen, Tianlong and Xu, Kaidi and Ding, Ying},
  journal={arXiv preprint arXiv:2502.14302},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
