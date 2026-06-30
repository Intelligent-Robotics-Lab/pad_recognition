import torch
import torch.nn as nn

class CrossModalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=16, num_layers=8, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,            # dimension of each modality embedding
            nhead=nhead,                # number of attention heads
            dim_feedforward=d_model*4,  # number of transformer encoder layers
            dropout=dropout,            # dropout probability to prevent overfitting
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Learned pooling instead of mean pooling
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, embeddings):
        x = self.transformer(embeddings)  # perform cross-modal attention
        weights = torch.softmax(self.attn_pool(x), dim=1)  # learn to weight modalities differently (B, 3, 1)
        fused = (weights * x).sum(dim=1)   # (B, d_model)
        
        return fused
