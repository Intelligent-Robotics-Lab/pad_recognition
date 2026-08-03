"""
Train script to test the text and audio modalaties together using the MELD dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features

from utils.dataloaders import get_meld_loaders
from models.emotion_model_text_audio import EmotionPADModelTA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Valid options: "mlp" and "transformer"
FUSION_TYPE = "transformer"

num_epochs = 50
learning_rate = 1e-4
seed = 42

torch.manual_seed(seed)

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

# Evaluate the model on an entire dataset split (using CCCs)
def evaluate(model, loader, name="VAL"):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            text = batch["text"][0]
            audio = batch["audio"][0]
            sample_rate = batch["sample_rate"][0]

            target = batch["pad"].to(device)

            text_feats = extract_text_features(text)
            audio_feats = extract_audio_features(audio, sample_rate)

            text_feats = torch.as_tensor(text_feats, dtype=torch.float32, device=device)
            audio_feats = torch.as_tensor(audio_feats, dtype=torch.float32, device=device)

            if text_feats.dim() == 2:
                text_feats = text_feats.unsqueeze(0)
            if audio_feats.dim() == 2:
                audio_feats = audio_feats.unsqueeze(0)

            pred = model(text_feats, audio_feats)

            preds_all.append(pred)
            targets_all.append(target)

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    ccc, avg = ccc_score(preds_all, targets_all)

    print(
        f"{name} CCC | "
        f"P: {ccc[0]:.4f}  "
        f"A: {ccc[1]:.4f}  "
        f"D: {ccc[2]:.4f}  "
        f"Avg: {avg:.4f}"
    )

    model.train()
    return avg

# batch_size=1 to match meld_collate's dict-of-lists shape.
train_loader, val_loader, _ = get_meld_loaders(batch_size=1)

model = EmotionPADModelTA(
    text_input_dim=1024,
    audio_input_dim=1024,
    d_model=512,
    fusion_type=FUSION_TYPE
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.SmoothL1Loss(reduction="none")

best_val_ccc = -float("inf")
os.makedirs("saved_models", exist_ok=True)

patience = 5
epochs_without_improvement = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        text = batch["text"][0]
        audio = batch["audio"][0]
        sample_rate = batch["sample_rate"][0]

        target = batch["pad"].to(device)

        if epoch == 0 and i == 0:
            print("First sample")
            print("Text:", text)
            print("Target:", target)

        text_feats = extract_text_features(text)
        audio_feats = extract_audio_features(audio, sample_rate)

        text_feats = torch.as_tensor(text_feats, dtype=torch.float32, device=device)
        audio_feats = torch.as_tensor(audio_feats, dtype=torch.float32, device=device)

        if text_feats.dim() == 2:
            text_feats = text_feats.unsqueeze(0)
        if audio_feats.dim() == 2:
            audio_feats = audio_feats.unsqueeze(0)

        optimizer.zero_grad()

        pred = model(text_feats, audio_feats)

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
    print(f"\nEpoch {epoch+1} Train MSE: {avg_loss:.4f}")

    val_ccc = evaluate(model, val_loader, "VAL")

    min_delta = 1e-3

    if val_ccc > best_val_ccc + min_delta:
        best_val_ccc = val_ccc
        epochs_without_improvement = 0

        save_path = os.path.join("saved_models", f"best_meld_ta_{FUSION_TYPE}.pth")
        torch.save(model.state_dict(), save_path)

        print(f"Saved new best model (VAL CCC = {best_val_ccc:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"No improvements for {epochs_without_improvement}/{patience} epochs.")

        if epochs_without_improvement >= patience:
            print("\nEarly stopping triggered.")
            break

print(f"\nTraining complete. Best validation CCC: {best_val_ccc:.4f}")
