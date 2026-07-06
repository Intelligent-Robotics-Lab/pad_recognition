import torch
import torch.nn as nn

class TextTransformerEncoder(nn.Module):
    """
    Learns a sequence representation from BERT token embeddings.
    Outputs a single [batch, d_model] vector using attention pooling.
    """
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

        self.pool_dropout = nn.Dropout(0.2)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2)
        )


    def forward(self, token_embeddings):

        x = self.transformer(token_embeddings)  # [B, T, H]
        x = nn.functional.layer_norm(x, x.shape[-1:]) # stabilize dynamics

        weights = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]

        pooled = (weights * x).sum(dim=1)  # [B, H]

        pooled = self.pool_dropout(pooled)  # Regularization
        pooled = self.proj(pooled)

        return pooled # [B, 512]

# Audio encoder using an LSTM model to capture temporal trends across emotion vectors
class AudioProjectionEncoder(nn.Module):
    def __init__(self, input_dim=1024, d_model=512, nhead=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.2
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attn = nn.Linear(d_model, 1)

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.2)
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
    def __init__(self, input_dim=7, d_model=512, use_gru=False):
        super().__init__()

        self.use_gru = use_gru

        self.input_proj = nn.Linear(input_dim, 128)
        self.input_dropout = nn.Dropout(0.2)

        rnn_hidden = 256

        if use_gru:
            self.rnn = nn.GRU(
                input_size=128,
                hidden_size=rnn_hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
        else:
            self.rnn = nn.LSTM(
                input_size=128,
                hidden_size=256, # Raised from 128 for experimentation
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 # Made slightly higher for experimentation, was originally 0.2
            )

        self.attn = nn.Linear(512, 1)

        self.output_proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2)
        )

        self.final_dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.input_dropout(self.input_proj(x))
        # h, _ = self.lstm(x)
        # w = torch.softmax(self.attn(h), dim=1)
        # pooled = (w * h).sum(dim=1)

        # out = self.output_proj(pooled)

        # Replaced for Experiment 2 - checking for attention collapse
        h, _ = self.rnn(x)
        pooled = h.mean(dim=1)
        pooled = self.output_proj(pooled)
        return pooled