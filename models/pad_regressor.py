import torch
import torch.nn as nn

class PADRegressors(nn.Module):
    # Simple regression heads for predicting Pleasure, Arousal, and Dominance from the fused embeddings we created
    def __init__(self, d_model=512, hidden_dim=None):
        """
        d_model: dimensionality of fused embeddings
        hidden_dim: optional hidden layer size (if you want a small MLP)
        """
        super().__init__()

        if hidden_dim is None:
            # Simple linear heads
            self.pleasure_head = nn.Linear(d_model, 1)
            self.arousal_head  = nn.Linear(d_model, 1)
            self.dominance_head = nn.Linear(d_model, 1)
        else:
            # Small MLP heads
            self.pleasure_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
            self.arousal_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
            self.dominance_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, fused_embedding: torch.Tensor):
        p = self.pleasure_head(fused_embedding)
        a = self.arousal_head(fused_embedding)
        d = self.dominance_head(fused_embedding)

        return p, a, d