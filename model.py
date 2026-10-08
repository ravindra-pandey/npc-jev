import torch
import torch.nn as nn
from transformers import AutoModel

class TaskHead(nn.Module):
    """Standard pointwise head for Noul and Score."""
    def __init__(self, hidden_size: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        return self.out_proj(self.dropout(self.activation(self.dense(x))))


class CrossOptionChoiceHead(nn.Module):
    """
    Listwise comparative choice head.
    Allows candidate option representations to attend to each other
    via multi-head self-attention before producing final logits.
    """
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Cross-option attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Scoring projection
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, option_embeddings):
        # option_embeddings shape: [M, hidden_size]
        # Reshape to [batch_size=1, seq_len=M, hidden_size] for attention
        x = option_embeddings.unsqueeze(0)
        
        # Self-attention across options with residual connection
        attn_out, _ = self.attn(query=x, key=x, value=x)
        x = self.norm(x + self.dropout(attn_out))
        
        # Squeeze back to [M, hidden_size] and project to scalar logits [M]
        x = x.squeeze(0)
        logits = self.out_proj(self.dropout(self.activation(self.dense(x)))).squeeze(-1)
        return logits


class DecisionEngineModel(nn.Module):
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Choice head with cross-option interaction
        self.choice_head = CrossOptionChoiceHead(hidden_size, num_heads=4, dropout=dropout)
        
        # Standard heads for Noul and Score
        self.noul_head = TaskHead(hidden_size, 1, dropout)
        self.score_head = TaskHead(hidden_size, 1, dropout)

    def forward(self, input_ids, attention_mask, task_types, candidate_splits):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # ModernBERT standard sequence pooling ([CLS] at index 0)
        cls_rep = outputs.last_hidden_state[:, 0, :]

        # Split flattened batch into per-item chunks
        split_sections = torch.split(cls_rep, candidate_splits, dim=0)
        
        predictions = []
        for sub_pooled, task in zip(split_sections, task_types):
            if task == "choice":
                # sub_pooled shape: [num_options, 768]
                # Outputs contrastive logits: [num_options]
                logits = self.choice_head(sub_pooled)
                predictions.append(("choice", logits))

            elif task == "noul":
                predictions.append(("noul", self.noul_head(sub_pooled).squeeze()))

            elif task == "score":
                predictions.append(("score", self.score_head(sub_pooled).squeeze(-1)))
                
        return predictions