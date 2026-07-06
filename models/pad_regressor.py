import torch
import torch.nn as nn

class PADRegressors(nn.Module):
    def __init__(self, d_model=512, hidden_dim=None):
        super().__init__()

        if hidden_dim is None: 
            # Simple linear heads
            self.pleasure_head = nn.Linear(d_model, 1)
            self.arousal_head  = nn.Linear(d_model, 1)
            self.dominance_head = nn.Linear(d_model, 1)

        else:
            # Expressive MLPs
            self.shared = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            )

            # Sepearate heads
            self.pleasure_head = nn.Linear(hidden_dim, 1)
            self.arousal_head = nn.Linear(hidden_dim, 1)
            self.dominance_head = nn.Linear(hidden_dim, 1)


    def forward(self, x):

        x = self.shared(x)

        p = self.pleasure_head(x)
        a = self.arousal_head(x)
        d = self.dominance_head(x)

        return p, a, d