import json
import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset

class MultiTaskDecisionDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class DecisionDataCollator:
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        flat_texts = []
        task_types = []
        candidate_splits = []
        targets = []

        for item in batch:
            kind = str(item.get("kind", "")).lower()
            context = str(item.get("context", "")).strip()
            question = str(item.get("question", "")).strip()
            options = item.get("options", [])
            target = item.get("target")

            if kind == "choice":
                for opt in options:
                    flat_texts.append(f"Context: {context} | Query: {question} | Action: {opt}")
                candidate_splits.append(len(options))
                task_types.append("choice")
                targets.append(int(target))

            elif kind == "score":
                for opt in options:
                    flat_texts.append(f"Context: {context} | Criterion: {question} | Tier: {opt}")
                candidate_splits.append(len(options))
                task_types.append("score")
                targets.append(int(target))

            elif kind == "noul":
                flat_texts.append(f"Context: {context} | Verification: {question}")
                candidate_splits.append(1)
                task_types.append("noul")
                targets.append(float(target))

        tokenized = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "task_types": task_types,
            "candidate_splits": candidate_splits,
            "targets": targets
        }