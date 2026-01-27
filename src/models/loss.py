import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.9):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, preds, targets):
        errors = targets - preds
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.abs(loss).mean()

class SmoothQuantileLoss(nn.Module):
    """
    Combination of Quantile Loss and Huber Loss for robustness.
    For errors < delta, use Huber (squared).
    For errors > delta, use Quantile (linear).
    """
    def __init__(self, quantile=0.9, delta=1.0):
        super(SmoothQuantileLoss, self).__init__()
        self.quantile = quantile
        self.delta = delta

    def forward(self, preds, targets):
        errors = targets - preds
        
        # Huber component
        abs_error = torch.abs(errors)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta).to(preds.device))
        linear = abs_error - quadratic
        huber_loss = 0.5 * quadratic**2 + self.delta * linear
        
        # Quantile weighting
        # If error > 0 (Underestimation for High), weight by q
        # If error < 0 (Overestimation), weight by 1-q
        weights = torch.where(errors > 0, self.quantile, 1 - self.quantile)
        
        loss = weights * huber_loss
        return loss.mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=4):
        super(MultiTaskLoss, self).__init__()
        self.num_tasks = num_tasks
        # Learnable log variances for homoscedastic uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        losses: List of [loss_high, loss_low, loss_sharpe, loss_dir]
        """
        dtype = losses[0].dtype
        device = losses[0].device
        
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + 0.5 * self.log_vars[i]
            
        return total_loss

def physics_constraint_loss(pred_high, current_price_ratio, big_buy_signal, threshold=0.5):
    """
    Enforce: If BigBuy > Threshold, PredHigh should be > CurrentPrice (Ratio > 1.0)
    Penalty if PredHigh <= 1.0
    """
    # big_buy_signal: [B]
    # pred_high: [B, 1] (Ratio)
    
    mask = big_buy_signal > threshold
    if not mask.any():
        return torch.tensor(0.0, device=pred_high.device)
        
    # We want pred_high > 1.01 (margin)
    target = 1.01
    violation = F.relu(target - pred_high[mask].squeeze())
    return violation.mean()

class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, preds, targets):
        # preds: [B, 1]
        # targets: [B]
        
        # Generate pairs
        # Simple approach: compare each element with the next one (shuffled batch)
        # Better: All pairs (B*B)
        
        # We'll use random permutation to make pairs efficiently
        # Shuffle indices
        idx = torch.randperm(preds.shape[0])
        p1 = preds
        p2 = preds[idx]
        t1 = targets
        t2 = targets[idx]
        
        # Target for MarginRankingLoss: 1 if p1 should be > p2 (i.e., t1 > t2), -1 otherwise
        target_sign = torch.sign(t1 - t2)
        
        # Filter out equal targets
        mask = target_sign != 0
        if not mask.any():
            return torch.tensor(0.0, device=preds.device)
            
        return self.loss_fn(p1[mask].squeeze(), p2[mask].squeeze(), target_sign[mask])

