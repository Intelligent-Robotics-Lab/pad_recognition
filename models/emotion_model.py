import torch
import torch.nn as nn
from models.encoders import TextTransformerEncoder, AudioProjectionEncoder, VideoProjectionEncoder
from models.fusion_model import CrossModalTransformer
from models.pad_regressor import PADRegressors

class EmotionPADModel(nn.Module):
    def __init__(self, text_input_dim, audio_input_dim, video_input_dim, d_model=512, use_gru=False):
        super().__init__()

        self.use_gru = use_gru

        self.text_encoder = TextTransformerEncoder(text_input_dim, d_model)
        self.audio_encoder = AudioProjectionEncoder(audio_input_dim, d_model, use_gru=self.use_gru)
        self.video_encoder = VideoProjectionEncoder(video_input_dim, d_model, use_gru=self.use_gru)

        self.fusion = CrossModalTransformer(
            d_model=d_model,
            nhead=8, 
            num_layers=4, 
            dropout=0.2
        )

        self.dropout = nn.Dropout(0.2)

        self.pad_regressor = PADRegressors(d_model=d_model, hidden_dim=256)

    def forward(self, text, audio, video):
        device = next(self.parameters()).device

        text_embedding = self.text_encoder(text)     # (B,512)
        audio_embedding = self.audio_encoder(audio)  # (B,512)
        video_embedding = self.video_encoder(video)  # (B,512)
        
        # Print the embeddings std deviation for debugging purposes
        print("Embedding stds:", text_embedding.std(), audio_embedding.std(), video_embedding.std())

        # Noise injection to prevent modality collapse and encourage robustness 
        if self.training:
            text_embedding = text_embedding + 0.01 * torch.randn_like(text_embedding)
            audio_embedding = audio_embedding + 0.01 * torch.randn_like(audio_embedding)
            video_embedding = video_embedding + 0.01 * torch.randn_like(video_embedding)

        # Stack the embedding for cross-modal transformer
        embeddings = torch.stack(
            [text_embedding, audio_embedding, video_embedding], 
            dim=1
        ) # (B, 3, 512)

        fused_embedding = self.fusion(embeddings)
        print(fused_embedding.std())

        # Add noise for robustness (may remove this and above noise at later stages, needs experimentation)
        if self.training:
            fused_embedding = fused_embedding + 0.01 * torch.randn_like(fused_embedding)

        fused_embedding = self.dropout(fused_embedding)

        p, a, d = self.pad_regressor(fused_embedding)

        return p, a, d