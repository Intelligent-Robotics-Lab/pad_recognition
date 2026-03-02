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
root_dir = r"/home/carter/pad_recognition/data/MELD.Raw"
batch_size = 64
num_epochs = 1  # mini training
d_model = 512   # embedding dimension

# Initialize and load dataset
dataset = MELDMultimodalDataset(root_dir=root_dir, split="train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initailize model and loss optimizer
model = EmotionPADModel(text_input_dim=776, audio_input_dim=126, video_input_dim=42, d_model=d_model).to(device)
model.train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop (mini run for testing)
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for text_batch, audio_paths, video_paths, pad_targets in dataloader:
        # Prep raw features tensors
        text_features_list = []
        audio_features_list = []
        video_features_list = []

        # Convert the raw text/audio/video inputs to embeddings, catching any errors and using zero vectors as fallbacks
        for i in range(len(text_batch)):
            # Text embeddings
            try:
                feat = prepare_text_features(text_batch[i])
                text_features_list.append(torch.tensor(feat, dtype=torch.float32, device=device))
            except Exception as e:
                print(f"[Warning] Text feature error at sample {i}: {e}")
                text_features_list.append(torch.zeros(776, device=device))  # assuming 776-dim text features

            # Audio embeddings
            try:
                feat = extract_audio_features(audio_paths[i])
                audio_features_list.append(torch.tensor(feat, dtype=torch.float32, device=device))
            except Exception as e:
                print(f"[Warning] Audio feature error at sample {i}: {e}")
                audio_features_list.append(torch.zeros(126, device=device))  # assuming 63-dim audio features

            # Video embeddings
            try:
                feat = extract_video_features(video_paths[i])
                video_features_list.append(torch.tensor(feat, dtype=torch.float32, device=device))
            except Exception as e:
                print(f"[Warning] Video feature error at sample {i}: {e}")
                video_features_list.append(torch.zeros(42, device=device))  # assuming 21-dim video features

        # Stack embeddings
        text_feats = torch.stack(text_features_list)
        audio_feats = torch.stack(audio_features_list)
        video_feats = torch.stack(video_features_list)

        # Convert PAD targets to tensor and move to device
        # Convert each target to float tensor and stack
        pad_targets_tensor = torch.stack([torch.tensor(t, dtype=torch.float32, device=device) for t in pad_targets])

        # Debug prints to verify shapes and data
        print("Text batch shape:", text_feats.shape, audio_feats.shape, video_feats.shape, pad_targets_tensor.shape)

        # Forward pass
        optimizer.zero_grad()
        try:
            pleasure, arousal, dominance = model(text_feats, audio_feats, video_feats)
        except Exception as e:
            print(f"[Error] Model forward failed: {e}")
            continue

        preds = torch.cat([pleasure, arousal, dominance], dim=1).squeeze(-1)  # shape: [batch, 3]

        # Debug: print shapes before loss calculation
        print("preds.shape:", preds.shape)
        print("pad_targets_tensor.shape:", pad_targets_tensor.shape)

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