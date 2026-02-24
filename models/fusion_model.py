import torch
import torch.nn as nn

class CrossModalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional pooling across modalities
        # self.pool = nn.Linear(d_model, d_model)

    def forward(self, embeddings):
        # Transformer expects (seq_len, batch_size, d_model), we have (batch_size, d_model) so add seq dim
        x = embeddings.transpose(0, 1)  # (n_modalities, batch_size, d_model)
        x = self.transformer(x)  # (n_modalities, batch_size, d_model)
        x = x.mean(dim=0)  # Pool across modalities: (batch_size, d_model)
        fused = x
        return fused