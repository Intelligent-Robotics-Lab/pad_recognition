import torch
import torch.nn as nn

from models.encoders import TextTransformerEncoder, AudioProjectionEncoder, VideoProjectionEncoder

from models.fusion import (MLPFusion, CrossModalTransformer)
from models.pad_regressor import PADRegressors

class EmotionPADModel(nn.Module):
    def __init__(self, text_hidden_dim=1024, audio_input_dim=1024, video_input_dim=1280, d_model=512, fusion_type="transformer"):
        super().__init__()

        self.text_encoder = TextTransformerEncoder(hidden_dim=text_hidden_dim, d_model=d_model)
        self.audio_encoder = AudioProjectionEncoder(input_dim=audio_input_dim, d_model=d_model)
        self.video_encoder = VideoProjectionEncoder(input_dim=video_input_dim, d_model=d_model)

        self.text_norm = nn.LayerNorm(d_model)
        self.audio_norm = nn.LayerNorm(d_model)
        self.video_norm = nn.LayerNorm(d_model) 

        self.modality_embeddings = nn.Parameter(torch.randn(3, d_model))

        if fusion_type == "mlp":
            self.fusion = MLPFusion(
                d_model
            )
        elif fusion_type == "transformer":
            self.fusion = CrossModalTransformer(
                d_model=d_model,
                nhead=1, 
                num_layers=1, 
                dropout=0.1
            )

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.pad_regressor = PADRegressors(d_model=d_model, hidden_dim=256)

    def forward(self, text, audio, video):

        text_embedding = self.text_norm(self.text_encoder(text))     # (B,512)
        audio_embedding = self.audio_norm(self.audio_encoder(audio))  # (B,512)
        video_embedding = self.video_norm(self.video_encoder(video))  # (B,512)
        
        embeddings = torch.stack([text_embedding, audio_embedding, video_embedding], dim=1) # (B, 3, 512)

        fused_embedding = self.fusion(embeddings)

        p, a, d = self.pad_regressor(fused_embedding)

        preds = torch.cat([p, a, d], dim=1)

        return preds