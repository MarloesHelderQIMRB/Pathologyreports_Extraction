# Pathologyreports_Extraction

Code for fine-tuning, evaluation, and inference of **LLaMA-3.1-8B-Instruct** to extract structured keratinocyte cancer information from unstructured pathology reports.

This repository accompanies the paper:

**Automated Identification of Keratinocyte Cancers in Pathology Reports Using Large Language Models**
*DOI: TBA*

## Overview

Pathology reports contain valuable clinical information, but are typically stored as free-text documents that are difficult to use for large-scale research. This repository provides the code used to fine-tune and evaluate a large language model (LLM) for automated extraction of information from skin pathology reports.

The model extracts structured information including:

* Diagnosis
* Anatomical site
* Facial subsite
* Lesion number

The resulting outputs are returned in a structured JSON format suitable for downstream epidemiological analyses and registry development.

## Base Model

This project uses the publicly available Meta LLM:

* LLaMA-3.1-8B-Instruct
* Hugging Face: `meta-llama/Llama-3.1-8B-Instruct`

The model was fine-tuned using parameter-efficient fine-tuning (LoRA).

## Repository Structure

```text
.
├── training.py      # Fine-tuning script
├── Inference.py     # Inference script for new pathology reports
├── LICENSE
└── README.md
```

## Requirements

Python 3.10+ is recommended.

Example installation:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac

# or

venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Dependencies include:

```bash
transformers
torch
datasets
peft
trl
accelerate
bitsandbytes
pandas
numpy
```

## Training

The training script performs supervised fine-tuning of LLaMA-3.1-8B-Instruct on pathology report data.

```bash
python training.py
```

Training data are not included in this repository because they contain sensitive clinical information and are subject to ethics restrictions.

## Inference

The inference script can be used to extract structured information from new pathology reports.

```bash
python Inference.py
```

The model returns structured JSON outputs that can be further processed for research analyses.

## Model Availability

The fine-tuned model was derived from data containing personal health information. Distribution of the model may require additional ethics approvals and institutional review. Therefore, only the code used for training and inference is currently provided.

## Citation

If you use this repository, please cite:

```text
Helder M, et al.
Automated Identification of Keratinocyte Cancers in Pathology Reports Using Large Language Models.
DOI: TBA
```

## License

This project is released under the Apache 2.0 License. See the LICENSE file for details.

## Contact

For questions regarding the code or manuscript, please open an issue on GitHub or contact the corresponding author.

