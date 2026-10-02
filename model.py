import torch
import torch.nn as nn
from transformers import AutoModel

class TaskHead(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        return self.out_proj(self.dropout(self.activation(self.dense(x))))

class DecisionEngineModel(nn.Module):
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.choice_head = TaskHead(hidden_size, 1, dropout)
        self.noul_head = TaskHead(hidden_size, 1, dropout)
        self.score_head = TaskHead(hidden_size, 1, dropout)

    def forward(self, input_ids, attention_mask, task_types, candidate_splits):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Masked Mean Pooling
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        # Split flattened batch back into per-sample chunks
        split_sections = torch.split(pooled, candidate_splits, dim=0)
        
        predictions = []
        for sub_pooled, task in zip(split_sections, task_types):
            if task == "choice":
                predictions.append(("choice", self.choice_head(sub_pooled).squeeze(-1)))
            elif task == "noul":
                predictions.append(("noul", self.noul_head(sub_pooled).squeeze()))
            elif task == "score":
                predictions.append(("score", self.score_head(sub_pooled).squeeze(-1)))
                
        return predictions