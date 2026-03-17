import torch
import torch.nn as nn

class PADRegressors(nn.Module):
    # Simple regression heads for predicting Pleasure, Arousal, and Dominance from the fused embeddings we created
    def __init__(self, d_model=512, hidden_dim=None):
        """
        Set of three independent regression heads for predicting the PAD dimensions from a
        fused multimodal embedding.

        P,A,D are correlated but not related so independent heads were chosen to allow the model to learn dimension-specific
        patterns and avoid negative transfer.

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
            # More expressive MLP heads
            self.pleasure_head = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
            self.arousal_head = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
            self.dominance_head = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            )

    def forward(self, fused_embedding: torch.Tensor):
        """
        Paramters:
            - fused_embedding : torch.Tensor
                Shape: (batch_size, d_model)
                Output of the cross-modal transformer
        Returns:
            - p, a, d : torch.Tensor
                Each of shape (batch_size, 1)
                Raw regression outputs
        """
        p = self.pleasure_head(fused_embedding)
        a = self.arousal_head(fused_embedding)
        d = self.dominance_head(fused_embedding)

        return p, a, d