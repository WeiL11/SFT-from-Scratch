import torch
import torch.nn.functional as F
import wandb
import numpy as np
import argparse
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """Tokenizes prompt and response, creating mask."""
    batch_input_ids = []
    batch_labels = []
    batch_response_masks = []
    max_len = 0
    
    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        o_ids = tokenizer.encode(o, add_special_tokens=False)
        full_ids = p_ids + o_ids
        mask = [0]*len(p_ids) + [1]*len(o_ids)
        
        batch_input_ids.append(torch.tensor(full_ids))
        batch_labels.append(torch.tensor(full_ids))
        batch_response_masks.append(torch.tensor(mask))
        max_len = max(max_len, len(full_ids))
        
    # Pad
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_input = torch.full((len(prompt_strs), max_len), pad_id, dtype=torch.long)
    padded_mask = torch.full((len(prompt_strs), max_len), 0, dtype=torch.long)
    
    for i in range(len(prompt_strs)):
        l = len(batch_input_ids[i])
        padded_input[i, :l] = batch_input_ids[i]
        padded_mask[i, :l] = batch_response_masks[i]
        
    return {
        "input_ids": padded_input[:, :-1],
        "labels": padded_input[:, 1:], # Shifted
        "response_mask": padded_mask[:, 1:] # Shifted
    }

def masked_mean(tensor, mask, dim=None):
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum().clamp(min=1)
    return masked.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)

def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    outputs = model(input_ids)
    logits = outputs.logits
    all_log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(all_log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    
    res = {"log_probs": log_probs}
    if return_token_entropy:
        probs = torch.exp(all_log_probs)
        entropy = -torch.sum(probs * all_log_probs, dim=-1)
        res["token_entropy"] = entropy
    return res

def sft_microbatch_train_step(policy_log_probs, response_mask, grad_acc_steps):
    """Computes SFT loss."""
    masked_log_probs = policy_log_probs * response_mask.to(policy_log_probs.dtype)
    loss = -masked_log_probs.sum() / response_mask.sum().clamp(min=1)
    scaled_loss = loss / grad_acc_steps
    scaled_loss.backward()
    return loss, scaled_loss

def log_generations(policy_model, tokenizer, prompts, ground_truths, step):
    """Generates responses using policy model and logs metrics (vLLM V1 compat)."""
    # 1. Generate using policy model (avoids vLLM weight sync API changes)
    device = next(policy_model.parameters()).device
    generated_text = []
    policy_model.eval()
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = policy_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            if "</answer>" in text:
                text = text.split("</answer>")[0] + "</answer>"
            generated_text.append(text)
    
    # 2. Score (Entropy)
    tokenized = tokenize_prompt_and_output(prompts, generated_text, tokenizer)
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    mask = tokenized["response_mask"].to(device)
    
    with torch.inference_mode():
        res = get_response_log_probs(policy_model, input_ids, labels, return_token_entropy=True)
        entropies = masked_mean(res["token_entropy"], mask, dim=1)
        
    # 3. Log
    rows = []
    scores = []
    for i, (p, gen, gt) in enumerate(zip(prompts, generated_text, ground_truths)):
        is_correct = 1.0 if gt in gen else 0.0 # Simple string match
        scores.append(is_correct)
        rows.append([p, gen, gt, is_correct, entropies[i].item()])
        
    metrics = {
        "eval/accuracy": np.mean(scores),
        "eval/avg_entropy": np.mean(entropies.cpu().numpy())
    }
    
    # Log Table
    if wandb.run:
        wandb.log(metrics, step=step)
        tbl = wandb.Table(columns=["Prompt", "Response", "GT", "Correct", "Entropy"])
        for r in rows: tbl.add_data(*r)
        wandb.log({"eval/generations": tbl}, step=step)
        
    print(f"Eval Step {step}: Acc={metrics['eval/accuracy']:.2%} Entropy={metrics['eval/avg_entropy']:.4f}")


# ==========================================
# 2. MAIN EXPERIMENT (Updated for Multi-GPU)
# ==========================================

def sft_experiment(train_gpu_id: int, eval_gpu_id: int, dataset_size: int = None):
    # Dynamic device strings based on arguments
    train_device = f"cuda:{train_gpu_id}"
    eval_device = f"cuda:{eval_gpu_id}"
    
    run_name = f"sft_size_{dataset_size if dataset_size else 'FULL'}_gpu{train_gpu_id}"
    
    # Initialize WandB with a unique name for this run
    wandb.init(
        project="sft_math_alignment", 
        name=run_name,
        config={
            "train_device": train_device,
            "eval_device": eval_device,
            "dataset_size": dataset_size,
            "model": "Qwen/Qwen2.5-Math-1.5B"
        }
    )
    
    print(f"🚀 Starting Run: {run_name}")
    print(f"   Training on: {train_device}")
    print(f"   Evaluating on: {eval_device}")
    
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Policy Model (Train)
    print("Loading Policy Model...")
    policy = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    ).to(train_device)
    
    # Data Loading
    print("Loading Data...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Handle Dataset Slicing
    if dataset_size is not None:
        print(f"Slice: Selecting first {dataset_size} examples.")
        dataset = dataset.select(range(min(len(dataset), dataset_size)))
    else:
        print("Slice: Using FULL dataset.")

    # Collate function
    def collate(batch):
        p = [f"User: {x['question']}\nAssistant: <think>" for x in batch]
        r = [f"{x['answer']} </answer>" for x in batch]
        return tokenize_prompt_and_output(p, r, tokenizer)
        
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    
    global_step = 0
    grad_acc_steps = 4
    policy.train()
    
    print("Starting Training Loop...")
    
    # We run for 2 epochs as a default for the experiment
    for epoch in range(2): 
        for batch in loader:
            input_ids = batch["input_ids"].to(train_device)
            mask = batch["response_mask"].to(train_device)
            labels = batch["labels"].to(train_device)
            
            # Forward Pass
            # Note: We compute logits manually here to feed into our custom microbatch step
            outputs = policy(input_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather log probs for the specific target tokens
            token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
            
            # Compute Loss & Backward (Microbatch Step)
            loss, _ = sft_microbatch_train_step(token_log_probs, mask, grad_acc_steps)
            
            # Optimizer Step (Gradient Accumulation)
            if (global_step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"train/loss": loss.item() * grad_acc_steps}, step=global_step)
                
            # Evaluation Loop (Every 50 steps)
            if global_step % 50 == 0:
                print(f"[{run_name}] Running Evaluation at step {global_step}...")
                
                val_data = load_dataset("gsm8k", "main", split="test").select(range(20))
                p = [f"User: {x['question']}\nAssistant: <think>" for x in val_data]
                gt = [x['answer'] for x in val_data]
                
                log_generations(policy, tokenizer, p, gt, global_step)
                
                # 4. Return to train mode
                policy.train()
                
            global_step += 1

    print(f"[{run_name}] Training Complete.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT Experiment on specific GPUs")
    
    # Argument for Training GPU ID
    parser.add_argument("--train_gpu", type=int, default=0, help="ID of the GPU to use for Training (Policy)")
    
    # Argument for Evaluation GPU ID
    parser.add_argument("--eval_gpu", type=int, default=1, help="ID of the GPU to use for vLLM Evaluation")
    
    # Argument for Dataset Size (Optional)
    parser.add_argument("--size", type=int, default=None, help="Number of examples to use (None = Full)")
    
    args = parser.parse_args()
    
    # Run the experiment with parsed arguments
    sft_experiment(args.train_gpu, args.eval_gpu, args.size)