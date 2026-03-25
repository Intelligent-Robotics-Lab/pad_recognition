import torch
import torch.nn as nn
from models.encoders import TextTransformerEncoder, AudioProjectionEncoder, VideoProjectionEncoder
from models.fusion_model import CrossModalTransformer
from models.pad_regressor import PADRegressors

class EmotionPADModel(nn.Module):
    """
    A multimodal architecture for predicting continuous PAD values from text, audio, and video
    """
    def __init__(self, text_input_dim, audio_input_dim, video_input_dim, d_model=512):
        super().__init__()
        # Modality-specific encoders to project raw inputs into a shared embedding space
        self.text_encoder = TextTransformerEncoder(text_input_dim, d_model)
        self.audio_encoder = AudioProjectionEncoder(audio_input_dim, d_model)
        self.video_encoder = VideoProjectionEncoder(video_input_dim, d_model)

        # Cross-modal fusion with transformer
        self.fusion = CrossModalTransformer(
            d_model=d_model,
            nhead=8, 
            num_layers=4, 
            dropout=0.1
        )

        # Regularization after fusion
        self.dropout = nn.Dropout(0.1)

        # PAD regression heads
        self.pad_regressor = PADRegressors(d_model=d_model, hidden_dim=256)

    def forward(self, text, audio, video):
        """
        Parameters:
            - text : torch.Tensor
                Shape: (B, text_dim)
            - audio : torch.Tensor
                Shape: (B, T_audio, audio_dim)
            - video : torch.Tensor
                Shape: (B, T_video, video_dim)
        Returns:
            - p, a, d : torch.Tensor
                Each of shape (B,1)
        """
        device = next(self.parameters()).device
        
        # Encode each modality
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

        # Cross-modal fusion
        fused_embedding = self.fusion(embeddings) # (B,512)
        print(fused_embedding.std()) # for debugging

        # Add optional noise for robustness
        if self.training:
            fused_embedding = fused_embedding + 0.01 * torch.randn_like(fused_embedding)

        # Dropout to regularize and encourage variation in the fused representation
        fused_embedding = self.dropout(fused_embedding)

        # Pad prediction
        p, a, d = self.pad_regressor(fused_embedding)

        # Apply tanh to constrain outputs to [-1,1]
        # p = torch.tanh(p)
        # a = torch.tanh(a)
        # d = torch.tanh(d)

        return p, a, d