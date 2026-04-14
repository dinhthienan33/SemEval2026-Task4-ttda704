import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTripletLoss(nn.Module):
    """
    Adaptive Triplet Loss with dynamic margin (Eq. 1).
    Harder samples get larger margins.
    """
    def __init__(self, base_margin=0.3, max_margin=1.0):
        super().__init__()
        self.base_margin = base_margin
        self.max_margin = max_margin

    def forward(self, anchor, pos, neg):
        pos_dist = (anchor - pos).pow(2).sum(1)
        neg_dist = (anchor - neg).pow(2).sum(1)

        difficulty = torch.clamp(neg_dist - pos_dist, min=0, max=self.max_margin - self.base_margin)
        adaptive_margin = self.base_margin + difficulty * 0.5

        loss = torch.relu(pos_dist - neg_dist + adaptive_margin)
        return loss.mean()


class MultiViewContrastiveLoss(nn.Module):
    """Contrastive loss for multi-view learning"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        sim_pos = torch.sum(anchor * positive, dim=1, keepdim=True) / self.temperature
        sim_neg = torch.sum(anchor * negative, dim=1, keepdim=True) / self.temperature

        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class SelfSupervisedViewAlignmentLoss(nn.Module):
    """
    Self-supervised alignment loss (Eq. 2):
    enforce that fusion of all views stays close to individual view embeddings.
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, theme_emb, plot_emb, outcome_emb, fusion_emb):
        view_avg = (theme_emb + plot_emb + outcome_emb) / 3.0

        view_avg = F.normalize(view_avg, p=2, dim=1)
        fusion_emb = F.normalize(fusion_emb, p=2, dim=1)

        loss = F.mse_loss(fusion_emb, view_avg)
        return self.weight * loss
