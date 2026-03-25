"""
Single-modality training script for debugging MELD PAD learning.

Purpose:
    Test whether each modality (text, audio, video) can learn PAD independently.

Key Features:
    - Uses existing encoders + PAD regressor
    - Early stopping on validation CCC
    - Debug prints for feature + prediction health
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR

from utils.helpers import PrecomputedDataset, multimodal_collate
from models.emotion_model import EmotionPADModel
from models.single_modality_model import SingleModalityModel

# Configuration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 50
learning_rate = 5e-4

# Change this line to test different modalities
MODALITY = "audio"   # "text", "audio", "video"

precomputed_dir = "data/MELD.Raw/precomputed_v2"

# Early stopping
patience = 10
best_val_ccc = -float("inf")
patience_counter = 0

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"best_{MODALITY}_model.pth")

# Loss calculations
def ccc_loss(pred, target):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)

    pred_var = pred.var(dim=0)
    target_var = target.var(dim=0)

    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)

    ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + 1e-8)
    return 1 - ccc.mean()

mse_loss = nn.MSELoss()

# Data loading
def load_split(prefix):
    files = sorted([
        os.path.join(precomputed_dir, f)
        for f in os.listdir(precomputed_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ])
    if not files:
        raise FileNotFoundError(f"No {prefix} files found in {precomputed_dir}")
    return ConcatDataset([PrecomputedDataset(f) for f in files])

train_dataset = load_split("train_features_")
val_dataset   = load_split("dev_features_")

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

# Model initialization
full_model = EmotionPADModel(
    text_input_dim=768,
    audio_input_dim=63,
    video_input_dim=52,
    d_model=512
).to(device)

if MODALITY == "text":
    encoder = full_model.text_encoder
elif MODALITY == "audio":
    encoder = full_model.audio_encoder
elif MODALITY == "video":
    encoder = full_model.video_encoder
else:
    raise ValueError("Invalid modality")

regressor = full_model.pad_regressor
model = SingleModalityModel(encoder, regressor).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Helper functions
def get_modality_batch(batch, modality):
    text, audio, video, target = batch

    if modality == "text":
        x = text
    elif modality == "audio":
        x = audio
    else:
        x = video

    return x, target

@torch.no_grad()
def evaluate(model, loader, modality):
    model.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        if batch is None:
            continue

        x, targets = get_modality_batch(batch, modality)

        x = x.to(device)
        targets = targets.to(device)

        preds = model(x)

        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    val_ccc = 1 - ccc_loss(all_preds, all_targets).item()

    model.train()
    return val_ccc

# Training loop
model.train()

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs} | Modality: {MODALITY}")

    running_loss = 0.0
    running_ccc = 0.0

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue

        x, targets = get_modality_batch(batch, MODALITY)

        x = x.to(device)
        targets = targets.to(device)

        # Debug input stats (first batch only)
        if epoch == 0 and batch_idx == 0:
            print("\n[DEBUG] Input stats:")
            print("  Shape:", x.shape)
            print("  Mean:", x.mean().item())
            print("  Std :", x.std().item())

        optimizer.zero_grad()

        preds = model(x)

        # For debugging, you can temporarily use ONLY MSE:
        loss_mse = mse_loss(preds, targets)
        loss_ccc = ccc_loss(preds, targets)
        loss = 0.5 * loss_mse + 0.5 * loss_ccc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        running_ccc += (1 - loss_ccc.item())

        # Debug prints
        if batch_idx % 20 == 0:
            print(f"\nBatch {batch_idx+1}")
            print(f"  Loss: {loss.item():.4f}")
            print("  Pred std:", preds.std(dim=0))

            for i in range(min(3, preds.size(0))):
                p, a, d = preds[i].tolist()
                tp, ta, td = targets[i].tolist()
                print(f"  Pred: [{p:.3f}, {a:.3f}, {d:.3f}] | "
                      f"Target: [{tp:.3f}, {ta:.3f}, {td:.3f}]")

    # ---- Epoch summary ----
    avg_loss = running_loss / len(train_loader)
    avg_ccc = running_ccc / len(train_loader)

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_loss:.4f}")
    print(f"  Train CCC : {avg_ccc:.4f}")

    # ---- Validation ----
    val_ccc = evaluate(model, val_loader, MODALITY)
    print(f"  Val CCC   : {val_ccc:.4f}")

    # ---- Early stopping ----
    if val_ccc > best_val_ccc:
        best_val_ccc = val_ccc
        patience_counter = 0

        torch.save(model.state_dict(), save_path)
        print("New best model saved.")

    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

    scheduler.step()

print("\nTraining complete.")
print(f"Best model saved at: {save_path}")