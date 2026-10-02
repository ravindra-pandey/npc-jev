import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from data_loader import MultiTaskDecisionDataset, DecisionDataCollator
from model import DecisionEngineModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Decision Engine on Validation Split")
    parser.add_argument("--val_file", type=str, default="datasets/decision_val_3k.jsonl", help="Validation JSONL file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/colab", help="Directory containing saved weights")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"EVALUATING DECISION ENGINE ON {args.val_file}")
    print(f"Device: {device} | Checkpoint: {args.checkpoint_dir}")
    print("=" * 80)

    if not os.path.exists(args.val_file):
        raise FileNotFoundError(f"Validation file '{args.val_file}' not found.")

    weights_path = os.path.join(args.checkpoint_dir, "decision_engine.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained model weights not found at '{weights_path}'.")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    model = DecisionEngineModel(args.model_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    val_dataset = MultiTaskDecisionDataset(args.val_file)
    collator = DecisionDataCollator(tokenizer, max_length=args.max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2
    )

    # Metric tracking
    metrics = {
        "choice": {"correct": 0, "total": 0},
        "noul": {"correct": 0, "total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "score": {"correct": 0, "total": 0, "absolute_errors": []}
    }

    pbar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            task_types = batch["task_types"]
            candidate_splits = batch["candidate_splits"]
            targets = batch["targets"]

            # Forward pass
            predictions = model(input_ids, attention_mask, task_types, candidate_splits)

            for (task, logits), target in zip(predictions, targets):
                if task == "choice":
                    pred_idx = logits.argmax(dim=-1).item()
                    if pred_idx == int(target):
                        metrics["choice"]["correct"] += 1
                    metrics["choice"]["total"] += 1

                elif task == "noul":
                    prob = torch.sigmoid(logits).item()
                    pred_label = 1.0 if prob >= 0.5 else 0.0
                    true_label = float(target)

                    if pred_label == true_label:
                        metrics["noul"]["correct"] += 1
                        if true_label == 1.0:
                            metrics["noul"]["tp"] += 1
                        else:
                            metrics["noul"]["tn"] += 1
                    else:
                        if pred_label == 1.0 and true_label == 0.0:
                            metrics["noul"]["fp"] += 1
                        else:
                            metrics["noul"]["fn"] += 1
                    metrics["noul"]["total"] += 1

                elif task == "score":
                    probs = F.softmax(logits, dim=-1)
                    k_indices = torch.arange(len(probs), dtype=torch.float, device=device)
                    expected_tier = torch.sum(probs * k_indices).item()
                    pred_discrete = probs.argmax(dim=-1).item()
                    true_tier = float(target)

                    if pred_discrete == int(true_tier):
                        metrics["score"]["correct"] += 1
                    metrics["score"]["absolute_errors"].append(abs(expected_tier - true_tier))
                    metrics["score"]["total"] += 1

    # --- Print Benchmark Report ---
    print("\n" + "=" * 80)
    print("FINAL VALIDATION BENCHMARK RESULTS")
    print("=" * 80)

    # 1. Choice Head Report
    c_tot = metrics["choice"]["total"]
    c_acc = (metrics["choice"]["correct"] / c_tot * 100) if c_tot > 0 else 0
    print(f"\n🎯 [CHOICE HEAD] (Physical & Social Action Selection)")
    print(f"  • Total Evaluated: {c_tot:,}")
    print(f"  • Top-1 Accuracy:  {c_acc:.2f}% (Random Baseline ~25-33%)")

    # 2. Noul Head Report
    n_tot = metrics["noul"]["total"]
    n_acc = (metrics["noul"]["correct"] / n_tot * 100) if n_tot > 0 else 0
    tp = metrics["noul"]["tp"]
    fp = metrics["noul"]["fp"]
    fn = metrics["noul"]["fn"]
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f"\n⚖️ [NOUL HEAD] (Boolean Rule & State Verification)")
    print(f"  • Total Evaluated: {n_tot:,}")
    print(f"  • Accuracy:        {n_acc:.2f}% (Random Baseline: 50.0%)")
    print(f"  • Precision:       {precision:.2f}%")
    print(f"  • Recall:          {recall:.2f}%")
    print(f"  • F1-Score:        {f1:.2f}%")

    # 3. Score Head Report
    s_tot = metrics["score"]["total"]
    s_acc = (metrics["score"]["correct"] / s_tot * 100) if s_tot > 0 else 0
    mae = sum(metrics["score"]["absolute_errors"]) / len(metrics["score"]["absolute_errors"]) if metrics["score"]["absolute_errors"] else 0
    print(f"\n📊 [SCORE HEAD] (Ordinal Rubric Scoring)")
    print(f"  • Total Evaluated: {s_tot:,}")
    print(f"  • Exact Tier Acc:  {s_acc:.2f}% (Random Baseline ~20-33%)")
    print(f"  • Mean Abs Error:  {mae:.3f} tiers off on average")

    # Overall Summary
    print("\n" + "=" * 80)
    avg_perf = (c_acc + n_acc + s_acc) / 3
    print(f"OVERALL ENGINE ACCURACY: {avg_perf:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()