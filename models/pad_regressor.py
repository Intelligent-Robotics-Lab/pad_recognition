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
            # More expressive MLP heads
            self.pleasure_head = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
            self.arousal_head = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
            self.dominance_head = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
            
            # # Look into replacement version seen below here: try and compare the differences
            # self.head = nn.Sequential(
            #     nn.Linear(d_model, 256),
            #     nn.GELU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(256, 128),
            #     nn.GELU(),
            #     nn.Linear(128, 3),
            #     nn.Tanh()
            # )

    def forward(self, fused_embedding: torch.Tensor):
        p = self.pleasure_head(fused_embedding)
        a = self.arousal_head(fused_embedding)
        d = self.dominance_head(fused_embedding)

        return p, a, d