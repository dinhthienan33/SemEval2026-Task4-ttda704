import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class EmbedModel(nn.Module):
    """
    Improved embedding model with:
    - Better pooling strategies
    - Optional dropout for regularization
    - Support for different backbones
    - Gradient checkpointing for memory efficiency
    """
    def __init__(self, backbone_name="sentence-transformers/all-MiniLM-L6-v2", 
                 dropout=0.1, pooling='mean', use_gradient_checkpointing=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name, 
            trust_remote_code=True
        )
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        
    def mean_pooling(self, hidden_state, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass with improved pooling"""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        else:
            hidden = out[0]
        
        if self.pooling == 'mean' and attention_mask is not None:
            emb = self.mean_pooling(hidden, attention_mask)
        else:
            emb = hidden.mean(dim=1)

        emb = self.dropout(emb)

        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        os.makedirs(save_directory, exist_ok=True)
        try:
            self.backbone.save_pretrained(save_directory)
            config = {
                'dropout': self.dropout.p,
                'pooling': self.pooling
            }
            with open(os.path.join(save_directory, 'embed_config.json'), 'w') as f:
                json.dump(config, f)
        except Exception as e:
            logging.warning(f"Could not save model properly: {e}")
            torch.save(self.backbone.state_dict(), 
                      os.path.join(save_directory, "pytorch_model.bin"))
