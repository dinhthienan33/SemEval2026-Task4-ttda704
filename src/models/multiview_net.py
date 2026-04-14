"""
Multi-view narrative encoder with projection heads and fusion layer.
Extracted from 23thang1/train_multiview_alignment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class MultiViewProjectionHead(nn.Module):
    """Separate projection heads for each view"""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()

        self.theme_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.plot_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.outcome_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, embeddings):
        """Project embeddings through each view head"""
        theme_proj = self.theme_head(embeddings)
        plot_proj = self.plot_head(embeddings)
        outcome_proj = self.outcome_head(embeddings)
        return theme_proj, plot_proj, outcome_proj


class MultiViewEncoder(nn.Module):
    """Encoder with shared backbone and view-specific projections"""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size
        self.projection_heads = MultiViewProjectionHead(self.embedding_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts, normalize=True):
        """Encode texts using backbone with mean pooling"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        model_output = self.model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )

        embeddings = self._mean_pooling(
            model_output.last_hidden_state,
            encoded["attention_mask"]
        )

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    def encode_views(self, theme_texts, plot_texts, outcome_texts):
        """Encode each view and project through respective heads"""
        theme_emb = self.encode(theme_texts, normalize=True)
        plot_emb = self.encode(plot_texts, normalize=True)
        outcome_emb = self.encode(outcome_texts, normalize=True)

        theme_proj = self.projection_heads.theme_head(theme_emb)
        plot_proj = self.projection_heads.plot_head(plot_emb)
        outcome_proj = self.projection_heads.outcome_head(outcome_emb)

        return {
            "theme": theme_emb,
            "plot": plot_emb,
            "outcome": outcome_emb,
            "theme_proj": theme_proj,
            "plot_proj": plot_proj,
            "outcome_proj": outcome_proj,
        }
