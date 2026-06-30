"""
Single-modality training script for debugging PAD learning on one modality at a time. This is useful for diagnosing issues with specific modalities 
and ensuring that the model can learn from each modality independently before combining them in the full multimodal model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
from utils.helpers import PrecomputedDataset, multimodal_collate, get_emotions_indices
from models.emotion_model import EmotionPADModel
from models.single_modality_model import SingleModalityModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 50
learning_rate = 1e-4 # Possibly change to 1e-4 specifically with audio test
use_gru = True

# Emotion to PAD mapping (inverse of emotion_to_label used earlier to calculate weights)
emotion_to_pad = torch.tensor([
    [-0.51,  0.59,  0.25],   # anger
    [-0.375, 0.1,   0.15],   # disgust
    [-0.64,  0.6,  -0.43],   # fear
    [ 0.4,   0.2,   0.1],    # joy
    [ 0.0,   0.0,   0.0],    # neutral
    [-0.34, -0.1,  -0.52],   # sadness
    [ 0.6,   0.6,   0.4],    # surprise
], device=device)

# Class counts from the train set to be used for setting weights in the loss function
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

MODALITY = "audio"   # "text", "audio", "video"

precomputed_dir = "data/MELD.Raw/precomputed_v4" # Update when newly computed
stats_path = os.path.join(precomputed_dir, "dataset_stats.pt")

patience = 10
best_val_ccc = -float("inf")
patience_counter = 0

best_train_ccc = None
best_epoch = -1

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"best_{MODALITY}_model.pth")

# Loss functions
def ccc_loss(pred, target):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)

    pred_var = pred.var(dim=0)
    target_var = target.var(dim=0)

    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)

    ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + 1e-8)
    ccc = torch.clamp(ccc, -1.0, 1.0)
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
    return ConcatDataset([PrecomputedDataset(f, stats_path=stats_path) for f in files])

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

full_model = EmotionPADModel(
    text_input_dim=1024,
    audio_input_dim=1024,
    video_input_dim=7,
    d_model=512,
    use_gru=use_gru
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

# # Experiment 1: freeze the encoder
# for param in encoder.parameters():
#     param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

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

        # Per sample MSE calculation
        mse = ((preds - targets) ** 2).mean(dim=1)

        # Emotion-based weighting
        emotion_idx = get_emotions_indices(targets)
        sample_weights = weights[emotion_idx]

        # Stablize weights to prevent extreme values
        sample_weights = torch.clamp(sample_weights, 0.2, 2.0)

        # Losses
        loss_ccc = ccc_loss(preds, targets)

        EXPERIMENT = 1

        if EXPERIMENT == 1:
            # Experiment 1 - Scheduled CCC + MSE
            alpha = min(1.0, epoch / 10)  # Linearly increase alpha from 0 to 1 over first 10 epochs
            loss = alpha * loss_ccc + (1 - alpha) * (sample_weights * mse).mean()

        # if EXPERIMENT == 2:
        #     # Varying the sheduling of CCC vs MSE
        #     alpha = min(1.0, epoch / 20)  # Linearly increase alpha from 0 to 1 over first 10 epochs
        #     loss = alpha * loss_ccc + (1 - alpha) * (sample_weights * mse).mean()

        # if EXPERIMENT == 3:
        #     # Varying the sheduling of CCC vs MSE
        #     alpha = min(1.0, (epoch / 15) ** 2)  # Linearly increase alpha from 0 to 1 over first 10 epochs
        #     loss = alpha * loss_ccc + (1 - alpha) * (sample_weights * mse).mean()

        # if EXPERIMENT == 2:
        #     # Experiment 2 - Pure MSE (with weighting)
        #     loss = (sample_weights * mse).mean()

        # if EXPERIMENT == 3:    
        #     # Experiment 3 - Fixed Hybrid (CCC + MSE)
        #     loss = 0.7 * loss_ccc + 0.3 * (sample_weights * mse).mean()

        # if EXPERIMENT == 4:
        #     # Experiment 4 - Pure CCC
        #     loss = loss_ccc

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
            print("  Weight mean:", sample_weights.mean().item())
            print("  Weight max :", sample_weights.max().item())

            for i in range(min(3, preds.size(0))):
                p, a, d = preds[i].tolist()
                tp, ta, td = targets[i].tolist()
                print(f"  Pred: [{p:.3f}, {a:.3f}, {d:.3f}] | "
                      f"Target: [{tp:.3f}, {ta:.3f}, {td:.3f}]")

    # Epoch summary
    avg_loss = running_loss / len(train_loader)
    avg_ccc = running_ccc / len(train_loader)

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_loss:.4f}")
    print(f"  Train CCC : {avg_ccc:.4f}")

    # Validation
    val_ccc = evaluate(model, val_loader, MODALITY)
    print(f"  Val CCC   : {val_ccc:.4f}")

    # Early stopping + model saving
    if val_ccc > best_val_ccc:
        best_val_ccc = val_ccc
        best_train_ccc = avg_ccc
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save(model.state_dict(), save_path)
        print("New best model saved.")

        # Debug best model state
        print("\n[BEST MODEL]")
        print(f"Epoch: {best_epoch}")
        print(f"Train CCC: {best_train_ccc:.4f}")
        print(f"Val CCC: {best_val_ccc:.4f}\n")

    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

    scheduler.step()

print("\nTraining complete.")
print(f"Best model saved at: {save_path}")

print("\nBest Model Summary")
print(f"Epoch: {best_epoch}")
print(f"Train CCC: {best_train_ccc}")
print(f"Val CCC: {best_val_ccc}")