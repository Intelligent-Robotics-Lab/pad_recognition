"""
Training script for the EmotionaPADModel on precomputed MELD features

Pipeline:
    - Load precomputed multimodal features (text, audio, video)
    - Batch using custom collate function
    - Forward pass through multimodal PAD model
    - Compute smooth L1 loss on p,a,d targets
    - Backdrop + optimzer setup
    - Save trained model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR

from utils.helpers import PrecomputedDataset, multimodal_collate
from models.emotion_model import EmotionPADModel 

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 10
d_model = 512
learning_rate = 5e-4

precomputed_dir = "data/MELD.Raw/precomputed"

# Load precomputed dataset
chunk_files = sorted([
    os.path.join(precomputed_dir, f)
    for f in os.listdir(precomputed_dir)
    if f.startswith("train_features_") and f.endswith(".pt")
])

if not chunk_files:
    raise FileNotFoundError(f"No precomputed chunks found in {precomputed_dir}")

datasets = [PrecomputedDataset(f) for f in chunk_files]
train_dataset = ConcatDataset(datasets)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=multimodal_collate
)

# Initialize model, loss, optimizer
model = EmotionPADModel(
    text_input_dim=768,
    audio_input_dim=63,
    video_input_dim=21,
    d_model=d_model
).to(device)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

model.train()

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            print(f"[Warning] Skipping batch {batch_idx} with invalid samples")
            continue

        text_feats, audio_feats, video_feats, pad_targets = batch

        # Move to device
        text_feats = text_feats.to(device)
        audio_feats = audio_feats.to(device)
        video_feats = video_feats.to(device)
        pad_targets = pad_targets.to(device)

        # Optional: Normalize features (better done during precomputing)
        text_feats = (text_feats - text_feats.mean(dim=0)) / (text_feats.std(dim=0) + 1e-6)
        audio_feats = (audio_feats - audio_feats.mean(dim=0)) / (audio_feats.std(dim=0) + 1e-6)
        video_feats = (video_feats - video_feats.mean(dim=0)) / (video_feats.std(dim=0) + 1e-6)

        optimizer.zero_grad()

        # Forward pass
        try:
            pleasure, arousal, dominance = model(text_feats, audio_feats, video_feats)
        except Exception as e:
            print(f"[Error] Model forward failed: {e}")
            continue

        preds = torch.cat([pleasure, arousal, dominance], dim=1)  # [batch, 3]

        # Loss computation
        loss = criterion(preds, pad_targets)
        loss.backward()
        optimizer.step()

        # Debug prints for mini run
        print(f"Batch {batch_idx+1} Loss: {loss.item():.4f}")
        for i in range(len(text_feats)):
            p, a, d = preds[i].tolist()
            tp, ta, td = pad_targets[i].tolist()
            print(f"Sample {i+1}: Pred P={p:.3f}, A={a:.3f}, D={d:.3f} | Target P={tp:.3f}, A={ta:.3f}, D={td:.3f}")
            print("Valid samples in batch:", len(text_feats))
            print("Prediction std:", preds.std(dim=0))

        # Step the schedular at the end of the epoch
        scheduler.step()

        #break  # remove this break for full epoch

# Save trained model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/emotionpad_trained.pth")

print("Training complete. Model saved to saved_models/emotionpad_trained.pth")