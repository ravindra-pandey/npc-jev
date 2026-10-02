import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskDecisionLoss(nn.Module):
    def __init__(self, device, ordinal_penalty_weight: float = 0.5):
        super().__init__()
        self.device = device
        self.ordinal_weight = ordinal_penalty_weight

    def forward(self, predictions, targets):
        total_loss = torch.tensor(0.0, device=self.device)
        stats = {"choice": 0.0, "noul": 0.0, "score": 0.0}
        
        for (task, logits), target in zip(predictions, targets):
            if task == "choice":
                target_t = torch.tensor(target, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(logits.unsqueeze(0), target_t.unsqueeze(0))
                total_loss += loss
                stats["choice"] += loss.item()

            elif task == "noul":
                target_t = torch.tensor(target, dtype=torch.float, device=self.device)
                loss = F.binary_cross_entropy_with_logits(logits, target_t)
                total_loss += loss
                stats["noul"] += loss.item()

            elif task == "score":
                target_t = torch.tensor(target, dtype=torch.long, device=self.device)
                ce_loss = F.cross_entropy(logits.unsqueeze(0), target_t.unsqueeze(0))
                
                # Ordinal distance penalty
                probs = F.softmax(logits, dim=0)
                k_indices = torch.arange(len(probs), dtype=torch.float, device=self.device)
                expected_tier = torch.sum(probs * k_indices)
                ord_loss = F.mse_loss(expected_tier, target_t.float())
                
                loss = ce_loss + (self.ordinal_weight * ord_loss)
                total_loss += loss
                stats["score"] += loss.item()

        # Normalize by batch item count
        batch_size = max(len(targets), 1)
        return total_loss / batch_size, {k: v / batch_size for k, v in stats.items()}