import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.helpers import MELDMultimodalDataset  # your MELD Dataset class
from models.emotion_model import EmotionPADModel       # your full multimodal model
from features.text_features import prepare_text_features
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = r"C:\Users\carte\emotion-recognition-hri\data\MELD.Raw"
batch_size = 3
num_epochs = 1  # mini training
d_model = 512   # embedding dimension

# Initialize and load dataset
dataset = MELDMultimodalDataset(root_dir=root_dir, split="train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initailize model and loss optimizer
model = EmotionPADModel(text_input_dim=776, audio_input_dim=63, video_input_dim=1280, d_model=d_model).to(device)
model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop (mini run for testing)
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for text_batch, audio_paths, video_paths, pad_targets in dataloader:
        # Prep embeddings lists
        text_embs_list = []
        audio_embs_list = []
        video_embs_list = []

        # Convert the raw text/audio/video inputs to embeddings, catching any errors and using zero vectors as fallbacks
        for i in range(len(text_batch)):
            # Text embeddings
            try:
                text_emb = torch.tensor(prepare_text_features(text_batch[i]), dtype=torch.float32, device=device).detach().clone()  # ensure no gradient tracking for raw features
            except Exception as e:
                print(f"[Warning] Text feature error at sample {i}: {e}")
                text_emb = torch.zeros(d_model, device=device)  # fallback
            text_embs_list.append(text_emb)

            # Audio embeddings
            try:
                audio_emb = torch.tensor(extract_audio_features(audio_paths[i]), dtype=torch.float32, device=device).detach().clone() 
            except Exception as e:
                print(f"[Warning] Audio feature error at sample {i}: {e}")
                audio_emb = torch.zeros(d_model, device=device)
            audio_embs_list.append(audio_emb)

            # Video
            try:
                video_emb = torch.tensor(extract_video_features(video_paths[i]), dtype=torch.float32, device=device).detach().clone()
            except Exception as e:
                print(f"[Warning] Video feature error at sample {i}: {e}")
                video_emb = torch.zeros(d_model, device=device)
            video_embs_list.append(video_emb)

        # Stack embeddings
        text_embs = torch.stack(text_embs_list)
        audio_embs = torch.stack(audio_embs_list)
        video_embs = torch.stack(video_embs_list)

        # Convert PAD targets to tensor and move to device
        # Convert each target to float tensor and stack
        pad_targets_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in pad_targets]).to(device)

        # Debug prints to verify shapes and data
        print("Text batch shape:", text_embs.shape, audio_embs.shape, video_embs.shape, pad_targets_tensor.shape)

        # Forward pass
        optimizer.zero_grad()
        try:
            pleasure, arousal, dominance = model(text_embs, audio_embs, video_embs)
        except Exception as e:
            print(f"[Error] Model forward failed: {e}")
            continue

        preds = torch.stack([pleasure, arousal, dominance], dim=1).squeeze(-1)  # shape: [batch, 3]
        # Uses MSE loss for regression of PAD values
        loss = criterion(preds, pad_targets_tensor)

        # Backpropogation computing gradients and updating weights
        loss.backward()
        optimizer.step()

        print("Batch Loss:", loss.item())
        print("Predicted PAD values:")
        for i in range(len(text_batch)):
            print(f"Sample {i+1}: P={preds[i,0].item():.3f}, A={preds[i,1].item():.3f}, D={preds[i,2].item():.3f}")
        break  # only first batch for mini test (remove this to train on all batches)

os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/emotionpad_trained.pth")
print ("Model saved to saved_models/emotionpad_trained.pth")

print("Mini training run complete.")