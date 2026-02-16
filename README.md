# Supervised Fine-Tuning from Scratch

Aligning Qwen2.5-Math-1.5B on GSM8K via SFT and Expert Iteration, using vLLM for sampling and Flash Attention for training.

## Key Accomplishments

- **Implemented** supervised fine-tuning (SFT) from scratch in an offline setting—built tokenization, loss masking, and training loops without relying on high-level frameworks
- **Resolved** multi-GPU orchestration to run training and inference on separate devices (e.g., train on GPU 0, evaluate with vLLM on GPU 1)
- **Aligned** Qwen2.5-Math to produce answers in a target format (e.g., `<think>...reasoning...</think>`), improving consistency and downstream usability

## Where to See Results (in the notebook)

| Notebook | What to look at |
|----------|-----------------|
| **Qwen_GSM8K_Run_GPUs.ipynb** | During training: `[Step N] Accuracy: X%, Format Score: Y%`. After a run: `Experiment complete. Model: ..., Val Acc: X%` |
| **Qwen_GSM8K_Tuned_GPUs.ipynb** | Each run: `→ Val Accuracy: X%`. Aggregate cell: `Best: run_id → X%` and best model path |

## Tech Stack

PyTorch · Transformers · vLLM · Flash Attention · WandB · GSM8K
