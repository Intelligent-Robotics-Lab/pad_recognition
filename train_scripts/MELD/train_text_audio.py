"""
Training script for the EmotionaPADModel, text and audio exclusive, on precomputed MELD features.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from utils.helpers import PrecomputedDataset, multimodal_collate, get_emotions_indices
from models.emotion_model_text_audio import EmotionPADModelTA

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 50
d_model = 512
learning_rate = 5e-4

precomputed_dir = "data/MELD.Raw/precomputed_v4"
stats_path = os.path.join(precomputed_dir, "dataset_stats.pt")

# Class counts (from the train set only, used for weighting)
counts = torch.tensor([
    1109,  # anger
    271,   # disgust
    268,   # fear
    1743,  # joy
    4710,  # neutral
    683,   # sadness
    1205   # surprise
], dtype=torch.float32, device=device)

# Inverse frequency weights
weights = counts.sum() / counts
weights = weights / weights.max()

# Loss functions
def ccc_loss(pred, target):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)

    pred_var = pred.var(dim=0)
    target_var = target.var(dim=0)

    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)

    ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + 1e-8)
    return 1 - ccc.mean()

# Load datasets
def load_split(prefix):
    files = sorted([
        os.path.join(precomputed_dir, f)
        for f in os.listdir(precomputed_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ])
    if not files:
        raise FileNotFoundError(f"No {prefix} files found")
    return ConcatDataset([PrecomputedDataset(f, stats_path=stats_path) for f in files])

train_dataset = load_split("train_features_")
val_dataset = load_split("dev_features_")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=multimodal_collate
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=multimodal_collate
)

model = EmotionPADModelTA(
    text_input_dim=1024,
    audio_input_dim=1024,
    d_model=d_model
).to(device)

mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        if batch is None:
            continue
        
        text_feats, audio_feats, _, pad_targets = batch

        text_feats = text_feats.to(device)
        audio_feats = audio_feats.to(device)
        pad_targets = pad_targets.to(device)

        pleasure, arousal, dominance = model(text_feats, audio_feats)
        preds = torch.cat([pleasure, arousal, dominance], dim=1)

        all_preds.append(preds)
        all_targets.append(pad_targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    val_ccc = 1 - ccc_loss(all_preds, all_targets).item()

    model.train()

    return val_ccc

model.train()

best_val_ccc = -float("inf")
patience = 10
patience_counter = 0

os.makedirs("saved_models", exist_ok=True)
save_path = "saved_models/best_model_ta.pth"

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    running_loss = 0.0
    running_ccc = 0.0

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            print(f"[Warning] Skipping batch {batch_idx} with invalid samples")
            continue

        text_feats, audio_feats, _, pad_targets = batch

        text_feats = text_feats.to(device)
        audio_feats = audio_feats.to(device)
        pad_targets = pad_targets.to(device)

        optimizer.zero_grad()

        try:
            pleasure, arousal, dominance = model(text_feats, audio_feats)
        except Exception as e:
            print(f"[Forward Error] Batch {batch_idx}: {str(e)}")
            raise

        preds = torch.cat([pleasure, arousal, dominance], dim=1)

        # Per sample MSE calculation
        mse = ((preds - pad_targets) ** 2).mean(dim=1)  # (batch,)

        # Emotion-based weighting
        emotion_idx = get_emotions_indices(pad_targets)
        sample_weights = weights[emotion_idx]

        # Stablize weights to prevent extreme values
        sample_weights = torch.clamp(sample_weights, 0.2, 2.0)

        # Losses
        loss_ccc = ccc_loss(preds, pad_targets)

        # Final weighted loss
        loss = loss_ccc + 0.2 * (sample_weights * mse).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_ccc += (1 - loss_ccc.item())

        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx+1} Loss: {loss.item():.4f}")
            print("Prediction std:", preds.std(dim=0))
            print("Weight mean:", sample_weights.mean().item())
            print("Weight max :", sample_weights.max().item())

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