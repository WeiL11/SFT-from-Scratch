# Supervised Fine-Tuning from Scratch

Aligning Qwen2.5-Math-1.5B on GSM8K via SFT and Expert Iteration, using vLLM for sampling and Flash Attention for training.

## Key Accomplishments

- **Implemented** supervised fine-tuning (SFT) from scratch in an offline setting—built tokenization, loss masking, and training loops without relying on high-level frameworks
- **Resolved** multi-GPU orchestration to run training and inference on separate devices (e.g., train on GPU 0, evaluate with vLLM on GPU 1)
- **Aligned** Qwen2.5-Math to produce answers in a target format (e.g., `<think>...reasoning...</think>`), improving consistency and downstream usability

## Where to See Results

After running SFT experiments:

| Location | Contents |
|----------|----------|
| `experiments/{run_name}/final_model/` | Trained model + tokenizer (~3 GB) |
| `experiments/{run_name}/run_info.json` | Validation accuracy, model path, config for that run |
| `experiments/tune_results/tune_summary_*.json` | Comparison of tuning runs; `best` field = best config and model path |

**Which model is best?**  
- Single run: open `experiments/{run_name}/run_info.json` → `val_accuracy`  
- Tuning runs: open `experiments/tune_results/tune_summary_*.json` → `best.model_path`, `best.val_accuracy`

## Tech Stack

PyTorch · Transformers · vLLM · Flash Attention · WandB · GSM8K
