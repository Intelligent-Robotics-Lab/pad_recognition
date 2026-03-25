import torch
import torch.nn as nn

class CrossModalTransformer(nn.Module):
    """
    A lightweight transformer encoder that perform *cross-modal attention*
    over three modality embeddings (text, audio, video) to produce a single fused embedding.

    Each modality is treated as a "token" in the transformer, allowing the model to learn complex interactions between modalities.

    This learns:
        - how modalities influence each other
        - how to weight modalities differently per sample
        - how to fuse complementary information into a single vector

    Expects: embeddings with shape (batch_size, 3, d_model)
    Outputs: single fused embedding with shape (batch_size, d_model)
    """
    def __init__(self, d_model=512, nhead=16, num_layers=8, dropout=0.1):
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
        """
        Embeddings shape: (batch_size, 3, 512)
        3 = [text, audio, video]
        """

        x = self.transformer(embeddings)  # perform cross-modal attention
        weights = torch.softmax(self.attn_pool(x), dim=1)  # learn to weight modalities differently (B, 3, 1)
        fused = (weights * x).sum(dim=1)   # (B, d_model)
        
        return fused
