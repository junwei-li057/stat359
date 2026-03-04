# Capability Repair and Trade-off Analysis in TinyStories LLM

This project investigates whether **capability repair** in small language models is competitive or trade-off driven.
Using a TinyStories-scale decoder-only transformer, we analyze whether fixing a specific failure (e.g., entity tracking) improves the target ability without degrading other capabilities.

The project compares **Full Fine-Tuning** and **LoRA Fine-Tuning** under a controlled patch dataset setting.

---

# Project Motivation

Small language models (SLMs) often exhibit localized failures such as:

* Repetition and low lexical diversity
* Broken entity tracking
* Structural inconspleteness
* Coherence degradation

A common solution is to construct a small **patch dataset** targeting one failure mode and fine-tune the model.
However, unlike large models, SLMs may face **capacity competition**, where improving one ability harms others.

This project aims to:

* Quantify capability trade-offs in SLMs
* Compare Full FT vs LoRA repair behavior
* Measure catastrophic forgetting
* Provide practical guidance for stable capability enhancement

---

# Project Structure

## 1. Tokenizer & Base Model

### `bpe_tokenizer.py`

Implements a custom BPE tokenizer with support for special tokens (`<user>`, `<assistant>`, etc.).

---

### `train_bpe_tokenizer_hf.py`

Trains the tokenizer on the TinyStories dataset.

**Usage:**
```bash
python scripts/train_bpe_tokenizer_hf.py --sample 10000
```

---

### `transformer_model.py`

Defines the TinyStories decoder-only transformer, including:

* Config class
* Self-attention
* Feed-forward layers
* Generation utilities

---

### `train_tinystories_model.py`

Trains the base TinyStories foundation model. 

**Usage:** -- held in Google Colab T4 GPU
```bash
python scripts/train_tinystories_model.py \
  --max_train_samples 200000 \
  --max_eval_samples 20000 \
  --batch_size 64 \
  --epochs 5 \
  --amp
```

Outputs:
* `results/models/tinystories_model/best_base_model.pth`

---

## 2. Patch Dataset Construction

### `build_entity_patch.py`

Creates a synthetic patch dataset focusing on **entity tracking repair**.

Features:
* 2–3 entity interactions
* Narrative consistency templates
* Ground-truth expected entities

**Usage:**
```bash
python scripts/generate_entity_patch.py \
    --output_path data/entity_patch.json
```

Output:
*  `data/entity_patch.json`

---

## 3. Capability Repair (Fine-Tuning) 

### `train_tinystories_model.py` (with patch added)

Performs **full fine-tuning** on the entity patch dataset.

**Usage:** -- held in Google Colab T4 GPU
```bash
python scripts/train_tinystories_model.py \
  --max_train_samples 200000 \
  --max_eval_samples 20000 \
  --batch_size 64 \
  --epochs 5 \
  --amp \
  --patch_file data/entity_patch.json
```

---

### `train_tinystories_LoRA_model.py`

Performs **LoRA repair** with frozen backbone weights.

**Usage:** -- held in Google Colab T4 GPU
```bash
python scripts/train_tinystories_LoRA_model.py \
  --max_train_samples 200000 \
  --max_eval_samples 20000 \
  --batch_size 64 \
  --epochs 5 \
  --lr 1e-4 \
  --amp \
  --patch_file data/entity_patch.json
```

Outputs:

* `results/models/tinystories_model/best_fullFT_model.pth`
* `results/models/tinystories_model/best_lora_model.pth`

---

## 4. Generation

### `generate_tinystories_text.py`

Generates outputs from the base, full FT, and LoRA model.

**Usage:**
```bash
python scripts/generate_tinystories_text.py \
    --model_path results/models/tinystories_model/best_base_model.pth \
    --eval_path data/eval_set.json \
    --output_path outputs/base_outputs.json \
    --max_length 80

python scripts/generate_tinystories_text.py \
    --model_path results/models/tinystories_model/best_fullFT_model.pth \
    --eval_path data/eval_set.json \
    --output_path outputs/base_fullFT_outputs.json \
    --max_length 80

python scripts/generate_lora_tinystories.py \
    --model_path results/models/tinystories_model/best_lora_model.pth \
    --eval_path data/eval_set.json \
    --output_path outputs/base_lora_outputs.json \
    --max_length 80
```

---

## 5. Evaluation

### `build_eval.py`

Generate a balanced synthetic evaluation set covering four failure categories (lexical diversity/repetition, entity tracking, structure, and coherence), allowing controlled and comparable capability measurement across models.

**Usage:**
```bash
python scripts/generate_eval_set.py
```

---

### `evaluate_metrics.py`

Computes automatic capability metrics (each with detailed help text):

* Lexical diversity
* Entity tracking
* Structure
* Coherence

**Usage:**
```bash
!python scripts/evaluate_metrics.py \
  --inputs outputs/base_outputs.json outputs/base_fullFT_outputs.json outputs/base_lora_outputs.json \
  --names base full_finetune lora \
  --output results/evaluation.csv
```

Output:
*  `results/evaluation.csv`

---

# Metrics Definition

### Lexical diversity

Measures token-level diversity as a proxy for repetition.
Higher values indicate fewer repetitive patterns.

### Entity tracking

Evaluates whether entities introduced in the prompt are:

* Recalled
* Balanced
* Free from hallucinated entities

### Structure

Measures narrative completeness and sentence regularity.

### Coherence

Approximates logical and semantic consistency across sentences.

---

# Requirements

* Python 3.8+
* PyTorch
* HuggingFace Datasets
* numpy
* tqdm
* tensorboard

---

# Notes

* All models use the custom TinyStories BPE tokenizer
* Metrics are automatic proxies (not human evaluation)
* Chat interface is optional and not required for experiments
