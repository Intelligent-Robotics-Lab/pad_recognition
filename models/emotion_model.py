# Was updated for doubel redundancy and to fix some bugs in the original code. The model now directly 
# takes the extracted features as input, projects them to a common dimension, and then fuses them using 
# a small transformer before regressing the PAD values. This should be more efficient and straightforward 
# for training on the MELD dataset.

import torch
import torch.nn as nn
from models.encoders import TextProjectionEncoder, AudioProjectionEncoder, VideoProjectionEncoder
from models.fusion_model import CrossModalTransformer
from models.pad_regressor import PADRegressors

class EmotionPADModel(nn.Module):
    def __init__(self, text_input_dim, audio_input_dim, video_input_dim, d_model=512):
        super().__init__()
        self.text_encoder = TextProjectionEncoder(text_input_dim, d_model)
        self.audio_encoder = AudioProjectionEncoder(audio_input_dim, d_model)
        self.video_encoder = VideoProjectionEncoder(video_input_dim, d_model)

        # Cross-modal fusion with small transformer
        self.fusion = CrossModalTransformer(d_model=d_model, nhead=4, num_layers=2, dropout=0.1)

        # PAD regression heads
        self.pad_regressor = PADRegressors(d_model=d_model, hidden_dim=256)

    def forward(self, text, audio, video):
        # Inputs are already numerical representations from feature extraction
        device = next(self.parameters()).device
        
        # Directly project to common dimension (no re-extraction needed)
        text_embedding = self.text_encoder(text)  # text: [batch_size, 768] → [batch_size, 512]
        audio_embedding = self.audio_encoder(audio)  # audio: [batch_size, 128] → [batch_size, 512]
        video_embedding = self.video_encoder(video)  # video: [batch_size, 1280] → [batch_size, 512]

        # Stack for fusion: [3, batch_size, 512] (n_modalities, batch, d_model)
        embeddings = torch.stack([text_embedding, audio_embedding, video_embedding], dim=0)

        # Fuse and regress
        fused_embedding = self.fusion(embeddings)
        p, a, d = self.pad_regressor(fused_embedding)
        return p, a, d