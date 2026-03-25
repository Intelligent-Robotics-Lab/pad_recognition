import torch
import torch.nn as nn

class SingleModalityModel(nn.Module):
    """
    Debug model for training on a single modality. Uses one encoder + existing PAD regressor.
    """
    def __init__(self, encoder, pad_regressor):
        super().__init__()
        self.encoder = encoder
        self.pad_regressor = pad_regressor

    def forward(self, x):
        # text  -> (B, 768), audio -> (B, T, 63), video -> (B, T, 52)

        # If temporal (audio/video), let encoder handle it
        embedding = self.encoder(x)  # (B, 512)

        # Debug: monitor embedding health
        if self.training:
            print("Embedding stats -> mean:",
                  embedding.mean().item(),
                  "std:",
                  embedding.std().item())

        p, a, d = self.pad_regressor(embedding)
        preds = torch.cat([p, a, d], dim=1)  # (B, 3)

        return preds