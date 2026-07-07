# LLM Playground - Build a small LLM from scratch on Colab T4

Educational project: pretrain + SFT + DPO + evaluation + chatbot for a small GPT-style LLM (~14M parameters), runnable on a free Colab T4 GPU.

## Architecture

GPT-style decoder-only Transformer built from scratch:

- **RMSNorm** instead of LayerNorm.
- **RoPE** (Rotary Position Embedding) instead of sinusoidal PE.
- **Multi-head causal self-attention** using `F.scaled_dot_product_attention` (Flash Attention).
- **Pre-LN residual blocks**: `x = x + attn(norm(x))`, `x = x + ffn(norm(x))`.
- **Weight tying** between token embedding and lm_head.

See `model/architecture.py` for details.

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Data Preparation

```bash
# Download ~300MB text subset from FineWeb
python data/scripts/download.py --target_mb 300 --output data/raw/corpus.txt

# Clean and deduplicate
python data/scripts/clean.py --input data/raw/corpus.txt --output data/processed/corpus_clean.txt
```

### 2. Train Tokenizer

```bash
python tokenizer/train_tokenizer.py --input data/processed/corpus_clean.txt --vocab_size 8192
```

### 3. Tokenize Corpus to .bin

```bash
python data/scripts/tokenize_corpus.py --input data/processed/corpus_clean.txt \
    --tokenizer tokenizer/vocab/tokenizer.json --output_dir data/processed
```

### 4. Pretrain

```bash
python training/pretrain.py --model_config configs/model_small.yaml \
    --train_config configs/train_pretrain.yaml
```

Resume training:

```bash
python training/pretrain.py --resume_from checkpoints/pretrain_step2000.pt
```

### 5. SFT (Supervised Fine-Tuning)

```bash
python training/sft.py --model_config configs/model_small.yaml \
    --train_config configs/train_sft.yaml
```

### 6. DPO (Direct Preference Optimization)

```bash
python training/rl/dpo.py --sft_checkpoint checkpoints/sft_final.pt
```

### 7. Evaluation

```bash
# Perplexity
python evaluation/perplexity.py --checkpoint checkpoints/pretrain_final.pt

# Multiple choice evaluation
python evaluation/multiple_choice_eval.py --checkpoint checkpoints/sft_final.pt
```

### 8. Chat

```bash
# CLI
python inference/chat.py --checkpoint checkpoints/sft_final.pt

# Web UI (Gradio)
python app/playground.py --checkpoint checkpoints/sft_final.pt
```

## Directory Structure

```
configs/     - hyperparameters
tokenizer/   - BPE tokenizer
model/       - Transformer architecture
data/        - data pipeline
training/    - pretrain, SFT, DPO
inference/   - generation & chat CLI
evaluation/  - perplexity & benchmarks
app/         - Gradio playground
notebooks/   - Colab notebooks
```
