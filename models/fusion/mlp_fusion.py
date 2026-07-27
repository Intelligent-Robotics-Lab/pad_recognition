import torch
import torch.nn as nn


class MLPFusion(nn.Module):
    # Simple concatenation and MLP fusion

    def __init__(self, d_model=512, num_modalities=2):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
        )

    def forward(self, embeddings):
        # embeddings : (B,2,512)

        x = embeddings.reshape(embeddings.size(0), -1)

        return self.fusion(x)