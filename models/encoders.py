import torch
import torch.nn as nn

class TextProjectionEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 512, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # 512-dim embedding, remove batch dim if input was single sample
        embedding = self.net(features)
        return embedding.squeeze(0)
    
class AudioProjectionEncoder(nn.Module):
    def __init__(self, input_dim, d_model=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, features: torch.Tensor):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        embedding = self.net(features)
        return embedding.squeeze(0)


class VideoProjectionEncoder(nn.Module):
    def __init__(self, input_dim, d_model=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, features: torch.Tensor):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        embedding = self.net(features)
        return embedding.squeeze(0)