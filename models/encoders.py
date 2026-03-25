import torch
import torch.nn as nn

# Started with a projection layer to map the 768-dim pooled BERT embedding down to 512
# Updated to accept embeddings of all tokens and used a small transformer to capture temporal dynamics across tokens
class TextTransformerEncoder(nn.Module):
    """
    Learns a sequence representation from BERT token embeddings.
    Outputs a single [batch, d_model] vector using attention pooling.
    """
    def __init__(self, hidden_dim=768, d_model=512, nhead=4, num_layers=2, dropout=0.1):
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

        # Attention pooling to learn token importance
        self.attn = nn.Linear(hidden_dim, 1)

        # Final projection to match the d_model
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, token_embeddings):
        """
        token_embeddings: [batch, seq_len, hidden_dim]
        """
        x = self.transformer(token_embeddings)  # [B, T, H]

        # Attention weights over tokens
        weights = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]

        # Take the weighted sum
        pooled = (weights * x).sum(dim=1)  # [B, H]

        return self.proj(pooled)  # [B, d_model]
    
# Audio encoder using an LSTM model to capture temporal trends across emotion vectors
class AudioProjectionEncoder(nn.Module):
    def __init__(self, input_dim=63, d_model=512):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, 1024)

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )

        self.output_proj = nn.Linear(1024, d_model)  # 512 due to bidirectional LSTM

    def forward(self, x):
        """
        x shape (B, T_audio, 63)
        Example batch (64, 150, 63)
        """

        x = self.input_proj(x)
        h, _ = self.lstm(x)
        pooled = h.mean(dim=1) # (B, 512)
        return self.output_proj(pooled)

# Video encoder using an LSTM model to capture temporal dynamics
class VideoProjectionEncoder(nn.Module):

    def __init__(self, input_dim=52, d_model=512):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, 1024)

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )

        self.output_proj = nn.Linear(1024, d_model)  # 512 due to bidirectional LSTM

    def forward(self,x):
        """
        x shape (b, T_video, 52)
        example batch (32, 40, 52)
        """

        x = self.input_proj(x)
        h, _ = self.lstm(x)
        pooled = h.mean(dim=1) # (B, 512)
        return self.output_proj(pooled)

# Former text encoder saved in case we want to use it for a simpler model or as a baseline
    # def __init__(self, input_dim: int, d_model: int = 512, dropout: float = 0.1):
    #     super().__init__()

    #     self.net = nn.Sequential(
    #         nn.Linear(input_dim, 1024),
    #         nn.GELU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(1024, 1024),
    #         nn.GELU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(1024, d_model),
    #         nn.LayerNorm(d_model)
    #     )

    # def forward(self, features: torch.Tensor) -> torch.Tensor:
    #     if features.dim() == 1:
    #         features = features.unsqueeze(0)

    #     # 512-dim embedding, remove batch dim if input was single sample
    #     embedding = self.net(features)
    #     return embedding.squeeze(0)