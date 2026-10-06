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
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts_a = []
        texts_b = []
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
                # Premise is context + query conditioned by task
                premise = f"[TASK: CHOICE] Context: {context} | Query: {question}"
                for opt in options:
                    texts_a.append(premise)
                    texts_b.append(str(opt).strip())
                candidate_splits.append(len(options))
                task_types.append("choice")
                targets.append(int(target))

            elif kind == "score":
                # Premise is context + criterion conditioned by task
                premise = f"[TASK: SCORE] Context: {context} | Criterion: {question}"
                for opt in options:
                    texts_a.append(premise)
                    texts_b.append(str(opt).strip())
                candidate_splits.append(len(options))
                task_types.append("score")
                targets.append(int(target))

            elif kind == "noul":
                # Premise is context, target is the assertion to verify
                premise = f"[TASK: NOUL] Context: {context}"
                texts_a.append(premise)
                texts_b.append(str(question).strip())
                candidate_splits.append(1)
                task_types.append("noul")
                targets.append(float(target))

        # Native pair tokenization with directional truncation
        tokenized = self.tokenizer(
            text=texts_a,
            text_pair=texts_b,
            padding=True,
            truncation="only_first",
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