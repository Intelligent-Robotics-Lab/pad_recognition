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

        self.fusion = CrossModalTransformer(
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dropout=0.2
        )

        self.dropout = nn.Dropout(0.2)

        self.pad_regressor = PADRegressors(d_model=d_model, hidden_dim=256)

    def forward(self, text, audio):

        text_embedding = self.text_encoder(text)     # (B, 512)
        audio_embedding = self.audio_encoder(audio)  # (B, 512)

        # Noise injection for robustness
        if self.training:
            text_embedding = text_embedding + 0.005 * torch.randn_like(text_embedding)
            audio_embedding = audio_embedding + 0.005 * torch.randn_like(audio_embedding)

        # Only 2 modalities now, but still use the same fusion model to learn cross-modal interactions
        embeddings = torch.stack(
            [text_embedding, audio_embedding],
            dim=1
        )  # (B, 2, 512)

        fused_embedding = self.fusion(embeddings)

        # Additional noise injection in training
        if self.training:
            fused_embedding = fused_embedding + 0.01 * torch.randn_like(fused_embedding)

        fused_embedding = self.dropout(fused_embedding)

        p, a, d = self.pad_regressor(fused_embedding)

        return p, a, d