"""
Train script for individual modalities (text and audio) using the IEMOCAP dataset
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features

from models.encoders import (TextTransformerEncoder, AudioProjectionEncoder)

from models.pad_regressor import PADRegressors

from models.single_modality_model import SingleModalityModel

from utils.dataloaders import get_iemocap_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODALITY = "audio"

num_epochs = 50
learning_rate = 1e-4
seed = 42

# PyTorch randomness can still affect other things so left it in
torch.manual_seed(seed)

print(f"Training single-modality PAD regressor on {MODALITY} features")

if MODALITY == "text":
    encoder = TextTransformerEncoder(hidden_dim=1024, d_model=512,)

elif MODALITY == "audio":
    encoder = AudioProjectionEncoder(input_dim=1024, d_model=512,)

else:
    raise ValueError("Only text and audio are currently supported.")

regressor = PADRegressors(d_model=512, hidden_dim=256,)

model = SingleModalityModel(encoder=encoder, pad_regressor=regressor).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.SmoothL1Loss(reduction="none")

# CCCs for evaluation only
@torch.no_grad()
def ccc_score(pred, target):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)

    pred_var = pred.var(dim=0, unbiased=False) + 1e-6
    target_var = target.var(dim=0, unbiased=False) + 1e-6

    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)

    ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + 1e-8)
    return ccc, ccc.mean().item()

# Load in the data
train_loader, val_loader, test_loader = get_iemocap_loaders(
    "data/iemocap.csv",
    batch_size=1
)

# Sanity check to confirm the splits
print(
    f"Train: {len(train_loader.dataset)} | "
    f"Val: {len(val_loader.dataset)} | "
    f"Test: {len(test_loader.dataset)}"
)

# Evaluate function (CCCs)
@torch.no_grad()
def evaluate(model, loader, name="VAL"):
    model.eval()

    preds_all = []
    targets_all = []

    for batch in loader:

        text = batch["text"][0]
        audio = batch["audio"][0]

        target = batch["pad"].to(device)

        if MODALITY == "text":
            feats = extract_text_features(text)

        elif MODALITY == "audio":
            sample_rate = batch["sample_rate"][0]
            feats = extract_audio_features(audio, sample_rate)

        feats = torch.tensor(
            feats,
            dtype=torch.float32,
            device=device
        )

        if feats.dim() == 2:
            feats = feats.unsqueeze(0)

        pred = model(feats)

        preds_all.append(pred)
        targets_all.append(target)

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)

    ccc, avg = ccc_score(preds_all, targets_all)

    print(
        f"{name} CCC | "
        f"P: {ccc[0]:.4f} "
        f"A: {ccc[1]:.4f} "
        f"D: {ccc[2]:.4f} "
        f"Avg: {avg:.4f}"
    )

    model.train()

    return avg

best_val_ccc = -float("inf")
os.makedirs("saved_models", exist_ok=True)

# Early stopping variables
patience = 3
epochs_without_improvement = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        text = batch["text"][0]
        audio = batch["audio"][0]
        target = batch["pad"].to(device)

        if epoch == 0 and i == 0:
            print("First sample")
            print("Text:", text)
            print("Target:", target)

        if MODALITY == "text":
            feats = extract_text_features(text)

        elif MODALITY == "audio":
            sample_rate = batch["sample_rate"][0]
            feats = extract_audio_features(audio, sample_rate)

        feats = torch.tensor(
            feats,
            dtype=torch.float32,
            device=device
        )

        # Depends on the extractor but just a precaution
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)

        optimizer.zero_grad()
        pred = model(feats)

        SmoothL1Loss = loss_fn(pred, target).mean(dim=1)
        loss = SmoothL1Loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

        if i % 200 == 0:
            print(f"\nSample {i}")

            print("Pred:", pred.detach().cpu())
            print("Target:", target.detach().cpu())

            print("Pred PAD mean:", pred.mean(dim=1).item())
            print("Target PAD mean:", target.mean(dim=1).item())

            print("Pred PAD std:", pred.squeeze(0).std().item())
            print("Target PAD std:", target.squeeze(0).std().item())

            print("Loss:", loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} Train Loss: {avg_loss:.4f}")

    val_ccc = evaluate(model, val_loader, "VAL")

    min_delta = 1e-3

    if val_ccc > best_val_ccc + min_delta:
        best_val_ccc = val_ccc
        epochs_without_improvement = 0

        save_path = os.path.join("saved_models", f"best_{MODALITY}_model_raw.pth")
        torch.save(model.state_dict(), save_path)
        
        print(
            f"Saved new best model "
            f"(VAL CCC = {best_val_ccc:.4f})"
        )

    else:
        epochs_without_improvement += 1
        print(f"No improvements for {epochs_without_improvement}/{patience} epochs.")

        if epochs_without_improvement >= patience:
            print("\nEarly stopping triggered")
            break

print("Training complete.")
print(f"Best validation CCC: {best_val_ccc:.4f}")