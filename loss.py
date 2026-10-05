import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskDecisionLoss(nn.Module):
    def __init__(self, device, ordinal_penalty_weight: float = 0.25):
        super().__init__()
        self.device = device
        self.ordinal_weight = ordinal_penalty_weight

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
                ce = F.cross_entropy(logits.unsqueeze(0), target_t.unsqueeze(0))
                
                probs = F.softmax(logits, dim=0)
                k_indices = torch.arange(len(probs), dtype=torch.float, device=self.device)
                expected_tier = torch.sum(probs * k_indices)
                ord_loss = F.mse_loss(expected_tier, target_t.float())
                
                # Scaled so score loss magnitude matches choice loss
                score_losses.append(ce + (self.ordinal_weight * ord_loss))

        # Average within each task first, then sum
        loss_choice = torch.stack(choice_losses).mean() if choice_losses else torch.tensor(0.0, device=self.device)
        loss_noul   = torch.stack(noul_losses).mean() if noul_losses else torch.tensor(0.0, device=self.device)
        loss_score  = torch.stack(score_losses).mean() if score_losses else torch.tensor(0.0, device=self.device)

        # Equal gradient contribution across all three active heads
        total_loss = loss_choice + loss_noul + loss_score

        stats = {
            "choice": loss_choice.item(),
            "noul": loss_noul.item(),
            "score": loss_score.item()
        }
        return total_loss, stats