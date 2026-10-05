import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskDecisionLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # Static normalization weights: brings each loss to ~1.0 at random init
        # Choice: 1 / ln(4) ≈ 0.72
        # Noul:   1 / ln(2) ≈ 1.44
        # Score:  1 / (ln(4) + 0.15) ≈ 0.65
        self.w_choice = 0.72
        self.w_noul   = 1.44
        self.w_score  = 0.65

    def forward(self, predictions, targets):
        choice_losses = []
        noul_losses = []
        score_losses = []
        
        for (task, logits), target in zip(predictions, targets):
            if task == "choice":
                target_t = torch.as_tensor(target, dtype=torch.long, device=self.device)
                choice_losses.append(F.cross_entropy(logits.unsqueeze(0), target_t.unsqueeze(0)))

            elif task == "noul":
                target_t = torch.as_tensor(target, dtype=torch.float, device=self.device)
                noul_losses.append(F.binary_cross_entropy_with_logits(logits, target_t))

            elif task == "score":
                target_t = torch.as_tensor(target, dtype=torch.long, device=self.device)
                num_tiers = logits.size(-1)
                
                # Standard Cross-Entropy
                ce_loss = F.cross_entropy(logits.unsqueeze(0), target_t.unsqueeze(0))
                
                # Normalized Ordinal Penalty (Bounded in [0, 1])
                probs = F.softmax(logits, dim=-1)
                k_indices = torch.arange(num_tiers, dtype=torch.float, device=self.device)
                expected_tier_norm = torch.sum(probs * k_indices) / max(num_tiers - 1, 1)
                target_tier_norm = target_t.float() / max(num_tiers - 1, 1)
                
                ord_loss_norm = F.mse_loss(expected_tier_norm, target_tier_norm)
                
                # Hybrid score loss
                score_losses.append(ce_loss + 0.25 * ord_loss_norm)

        # Intra-task means
        l_choice = torch.stack(choice_losses).mean() if choice_losses else None
        l_noul   = torch.stack(noul_losses).mean() if noul_losses else None
        l_score  = torch.stack(score_losses).mean() if score_losses else None

        # Dynamically average ONLY over the tasks present in this specific batch
        active_losses = []
        stats = {"choice": 0.0, "noul": 0.0, "score": 0.0}

        if l_choice is not None:
            active_losses.append(self.w_choice * l_choice)
            stats["choice"] = l_choice.item()
        if l_noul is not None:
            active_losses.append(self.w_noul * l_noul)
            stats["noul"] = l_noul.item()
        if l_score is not None:
            active_losses.append(self.w_score * l_score)
            stats["score"] = l_score.item()

        # Mean over active tasks in batch ensures batches with 2 tasks don't drop in magnitude vs 3 tasks
        total_loss = torch.stack(active_losses).mean()

        return total_loss, stats