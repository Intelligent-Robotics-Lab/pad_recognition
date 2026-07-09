import torch
import torch.nn as nn

class TextTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim=1024, d_model=512, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attn = nn.Linear(hidden_dim, 1)

        self.pool_dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )


    def forward(self, token_embeddings):

        x = self.transformer(token_embeddings)  # [B, T, H]
        x = nn.functional.layer_norm(x, x.shape[-1:]) # stabilize dynamics

        weights = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]

        pooled = (weights * x).sum(dim=1)  # [B, H]

        pooled = self.pool_dropout(pooled)  # Regularization
        pooled = self.proj(pooled)

        return pooled # [B, 512]

# Audio encoder using a transformer model to capture temporal trends across emotion vectors
class AudioProjectionEncoder(nn.Module):
    def __init__(self, input_dim=1024, d_model=512, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attn = nn.Linear(d_model, 1)

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )


    def forward(self, x): # x: [B, T, 1024]
        
        x = self.input_proj(x)

        x = self.transformer(x)

        x = nn.functional.layer_norm(x, x.shape[-1:])

        weights = torch.softmax(self.attn(x), dim=1)

        pooled = (weights * x).sum(dim=1)

        return self.out(pooled)  # [B, 512]

# Video encoder using an LSTM model to capture temporal dynamics
class VideoProjectionEncoder(nn.Module):
    def __init__(self, input_dim=1280, d_model=512, nhead=4, num_layers=2, dropout = 0.2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.attn = nn.Linear(d_model, 1)

        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, T, 1280]

        x = self.input_proj(x)

        x = self.transformer(x)

        x = nn.functional.layer_norm(x, x.shape[-1:])

        weights = torch.softmax(self.attn(x), dim=1)

        pooled = (weights * x).sum(dim=1)

        return self.output(pooled)