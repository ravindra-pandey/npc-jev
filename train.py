import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from data_loader import MultiTaskDecisionDataset, DecisionDataCollator
from model import DecisionEngineModel
from loss import MultiTaskDecisionLoss

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

BASE_DATA_DIR ='datasets'

def get_profile_config(profile: str):
    """Production configuration router."""
    if profile == "debug":
        return {
            "train_file": "debug_sample.jsonl",
            "epochs": 1,
            "max_steps": 50,
            "batch_size": 4,          # Fits in 4GB VRAM
            "grad_accum": 2,          # Effective batch = 8
            "fp16": True,
            "use_8bit_adam": True
        }
    elif profile == "colab":
        return {
            "train_file": "decision_train_44k.jsonl",
            "epochs": 3,
            "max_steps": None,
            "batch_size": 16,         # Scales to T4/A100 (16GB+ VRAM)
            "grad_accum": 4,          # Effective batch = 64
            "fp16": True,
            "use_8bit_adam": False    # Not strictly needed on 16GB VRAM
        }
    else:
        raise ValueError("Profile must be 'debug' or 'colab'")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, required=True, choices=["debug", "colab"], help="Runtime environment profile.")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    config = get_profile_config(args.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(f"STARTING PRODUCTION RUN | PROFILE: [{args.profile.upper()}]")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DecisionEngineModel(args.model_name).to(device)
    loss_fn = MultiTaskDecisionLoss(device)

    dataset = MultiTaskDecisionDataset(f"{BASE_DATA_DIR}/{config['train_file']}")
    collator = DecisionDataCollator(tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True)

    if config["use_8bit_adam"] and HAS_BNB and torch.cuda.is_available():
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = (len(loader) // config["grad_accum"]) * config["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=config["fp16"])

    model.train()
    global_step = 0

    for epoch in range(config["epochs"]):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            with torch.amp.autocast('cuda', enabled=config["fp16"]):
                predictions = model(input_ids, attention_mask, batch["task_types"], batch["candidate_splits"])
                loss, stats = loss_fn(predictions, batch["targets"])
                loss = loss / config["grad_accum"]

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config["grad_accum"] == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                pbar.set_postfix({"loss": f"{loss.item() * config['grad_accum']:.4f}"})

                if config["max_steps"] and global_step >= config["max_steps"]:
                    print("\n[Debug Mode] Max steps reached. Halting.")
                    break

        if config["max_steps"] and global_step >= config["max_steps"]:
            break

    # Save artifact
    out_path = os.path.join(args.output_dir, args.profile)
    os.makedirs(out_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_path, "decision_engine.pt"))
    tokenizer.save_pretrained(out_path)
    print(f"\n✓ Engine saved to {out_path}")

if __name__ == "__main__":
    train()