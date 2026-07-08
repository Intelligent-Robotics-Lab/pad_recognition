import torch
import torch.nn as nn

"""
The original transformer architecture (2 layers, 4 heads, lage feed-forward network was overparamterized for the small token fusion problem.
Simpler learned fusion methods preserve complementary informatio nbetter while lightweight transformers remain viable but require further tuning.
"""

class CrossModalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=1, num_layers=1, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,            # dimension of each modality embedding
            nhead=nhead,                    # number of attention heads
            dim_feedforward=d_model,    # number of transformer encoder layers
            dropout=dropout,                # dropout probability to prevent overfitting
            activation='gelu',
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.output_projection = nn.Linear(d_model * 2, d_model,)

    def forward(self, embeddings):
        # Embeddings: [B, 2, 512]

        x = nn.functional.layer_norm(embeddings, embeddings.shape[-1:],)

        x = self.transformer(x)

        # [B, 512] -> [B, 1024]
        x = x.reshape(x.size(0), -1)

        # [B, 1024] -> [B, 512]
        x = self.output_projection(x)

        return x