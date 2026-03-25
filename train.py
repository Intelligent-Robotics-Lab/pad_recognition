"""
Training script for the EmotionaPADModel on precomputed MELD features

Pipeline:
    - Load precomputed multimodal features (text, audio, video)
    - Batch using custom collate function
    - Forward pass through multimodal PAD model
    - Compute smooth L1 loss on p,a,d targets
    - Backprop + optimizer setup
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
num_epochs = 50
d_model = 512
learning_rate = 5e-4

precomputed_dir = "data/MELD.Raw/precomputed_v2"

# Loss
def ccc_loss(pred, target):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)

    pred_var = pred.var(dim=0)
    target_var = target.var(dim=0)

    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)

    ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + 1e-8)
    return 1 - ccc.mean()

# Load train dataset
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

# Validation dataset
val_files = sorted([
    os.path.join(precomputed_dir, f)
    for f in os.listdir(precomputed_dir)
    if f.startswith("dev_features_") and f.endswith(".pt")
])

if not val_files:
    raise FileNotFoundError(f"No validation chunks found in {precomputed_dir}")

val_datasets = [PrecomputedDataset(f) for f in val_files]
val_dataset = ConcatDataset(val_datasets)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=multimodal_collate
)

# Evaluation
@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    for batch in val_loader:
        if batch is None:
            continue
        
        text_feats, audio_feats, video_feats, pad_targets = batch

        text_feats = text_feats.to(device)
        audio_feats = audio_feats.to(device)
        video_feats = video_feats.to(device)
        pad_targets = pad_targets.to(device)

        pleasure, arousal, dominance = model(text_feats, audio_feats, video_feats)
        preds = torch.cat([pleasure, arousal, dominance], dim=1)

        all_preds.append(preds)
        all_targets.append(pad_targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    val_ccc = 1 - ccc_loss(all_preds, all_targets).item()

    model.train()
    return val_ccc

# Model
model = EmotionPADModel(
    text_input_dim=768,
    audio_input_dim=63,
    video_input_dim=52,
    d_model=d_model
).to(device)

mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

model.train()

# Early stopping setup
best_val_ccc = -float("inf")
patience = 5
patience_counter = 0

# Create save directory ONCE
os.makedirs("saved_models", exist_ok=True)
save_path = "saved_models/best_model.pth"

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    running_loss = 0.0
    running_ccc = 0.0

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            print(f"[Warning] Skipping batch {batch_idx} with invalid samples")
            continue

        text_feats, audio_feats, video_feats, pad_targets = batch

        text_feats = text_feats.to(device)
        audio_feats = audio_feats.to(device)
        video_feats = video_feats.to(device)
        pad_targets = pad_targets.to(device)

        optimizer.zero_grad()

        try:
            pleasure, arousal, dominance = model(text_feats, audio_feats, video_feats)
        except Exception as e:
            print(f"[Forward Error] Batch {batch_idx}: {str(e)}")
            raise

        preds = torch.cat([pleasure, arousal, dominance], dim=1)

        loss_mse = mse_loss(preds, pad_targets)
        loss_ccc = ccc_loss(preds, pad_targets)

        loss = 0.5 * loss_mse + 0.5 * loss_ccc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_ccc += (1 - loss_ccc.item())

        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx+1} Loss: {loss.item():.4f}")
            print("Prediction std:", preds.std(dim=0))

    avg_loss = running_loss / len(train_loader)
    avg_ccc = running_ccc / len(train_loader)

    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f} | Avg CCC: {avg_ccc:.4f}")

    # Validation + Early Stopping
    val_ccc = evaluate(model, val_loader, device)
    print(f"Validation CCC: {val_ccc:.4f}")

    if val_ccc > best_val_ccc:
        best_val_ccc = val_ccc
        patience_counter = 0

        torch.save(model.state_dict(), save_path)
        print("New best model saved.")

    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    scheduler.step()

print(f"Training complete. Best model saved to {save_path}")