import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=4):
        super(MultiTaskLoss, self).__init__()
        self.num_tasks = num_tasks
        # Learnable log variances for homoscedastic uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        losses: torch.Tensor of shape [num_tasks], each element is a scalar task loss.
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + 0.5 * self.log_vars[i]
        return total_loss


class PeakDayLoss(nn.Module):
    def __init__(self, sigma=2.0):
        super(PeakDayLoss, self).__init__()
        self.sigma = sigma
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits, target_idx):
        length = logits.shape[1]
        device = logits.device
        idx = torch.arange(length, device=device).unsqueeze(0)
        target_idx = target_idx.unsqueeze(1)
        dist = (idx - target_idx).float()
        soft = torch.exp(-0.5 * (dist / self.sigma) ** 2)
        soft = soft / soft.sum(dim=1, keepdim=True)
        log_probs = torch.log_softmax(logits, dim=1)
        return self.kl(log_probs, soft)
