import torch
import torch.nn as nn
from models.encoders import TextTransformerEncoder, AudioProjectionEncoder
from models.fusion_model import CrossModalTransformer
from models.pad_regressor import PADRegressors

class EmotionPADModelTA(nn.Module):
    """
    Text + Audio only version for PAD prediction
    """
    def __init__(self, text_input_dim, audio_input_dim, d_model=512):
        super().__init__()

        self.text_encoder = TextTransformerEncoder(text_input_dim, d_model)
        self.audio_encoder = AudioProjectionEncoder(audio_input_dim, d_model)

        self.text_norm = nn.LayerNorm(d_model)
        self.audio_norm = nn.LayerNorm(d_model)

        self.fusion = CrossModalTransformer(
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dropout=0.2
        )

        # Learnable embeddings to tell the transformer which modality each token is
        self.modality_embeddings = nn.Parameter(torch.randn(2, d_model))

        self.pad_regressor = PADRegressors(d_model=d_model, hidden_dim=256)

    def forward(self, text, audio):

        # Normalize encoder outputs so both modalities have similar feature statistics
        text_embedding = self.text_norm(self.text_encoder(text))     # (B, 512)
        audio_embedding = self.audio_norm(self.audio_encoder(audio))  # (B, 512)

        text_embedding = text_embedding + self.modality_embeddings[0]
        audio_embedding = audio_embedding + self.modality_embeddings[1]

        # Only 2 modalities now, but still use the same fusion model to learn cross-modal interactions
        embeddings = torch.stack(
            [text_embedding, audio_embedding],
            dim=1
        )  # (B, 2, 512)

        fused_embedding = self.fusion(embeddings)

        p, a, d = self.pad_regressor(fused_embedding)

        preds = torch.cat([p, a, d], dim=1) # (B, 3)

        return preds