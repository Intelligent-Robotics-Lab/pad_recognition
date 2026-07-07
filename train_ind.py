"""
Train script for individual modalities (text and audio) using the IEMOCAP dataset
"""

import os
import random
from io import BytesIO

import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Audio, load_dataset

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from models.emotion_model import EmotionPADModel
from models.single_modality_model import SingleModalityModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODALITY = "audio"

num_epochs = 10
learning_rate = 3e-5
seed = 42
use_gru = False

random.seed(seed)
torch.manual_seed(seed)

print(f"Training single-modality PAD regressor on {MODALITY} features")

# Full model to pull in the encoder sub-modules (will change in future iterations)
full_model = EmotionPADModel(
    text_input_dim=1024,
    audio_input_dim=1024,
    video_input_dim=7,
    d_model=512,
    use_gru=use_gru,
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

def ccc_loss(pred, target):
    pred = torch.nan_to_num(pred)
    target = torch.nan_to_num(target)

    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    pred_var = pred.var(dim=0) + 1e-6
    target_var = target.var(dim=0) + 1e-6

    cov = (pred * target).mean(dim=0)

    ccc = (2 * cov) / (pred_var + target_var + 1e-6)

    return 1 - ccc.mean()

def build_features(sample):
    if MODALITY == "text":
        feats = extract_text_features(sample["transcription"])
    elif MODALITY == "audio":
        waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))
        feats = extract_audio_features(waveform, sr)
    else:
        raise ValueError("Video is not supported in this single-modality script")

    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)


print("Loading IEMOCAP...")
ds = load_dataset("AbstractTTS/IEMOCAP")["train"]
ds = ds.cast_column("audio", Audio(decode=False))

print("Dataset size:", len(ds))

all_targets = torch.tensor(
    [[ds[i]["EmoVal"], ds[i]["EmoAct"], ds[i]["EmoDom"]] for i in range(len(ds))],
    dtype=torch.float32,
)
mu = all_targets.mean(dim=0).to(device)

indices = list(range(len(ds)))
random.shuffle(indices)

train_split = int(0.8 * len(indices))
val_split = int(0.9 * len(indices))

train_idx = indices[:train_split]
val_idx = indices[train_split:val_split]
test_idx = indices[val_split:]

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# Evaluate function (CCCs)
def evaluate(model, indices, name="VAL"):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for i in indices:
            sample = ds[i]
            target = torch.tensor(
                [sample["EmoVal"], sample["EmoAct"], sample["EmoDom"]],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            feats = build_features(sample)
            pred = model(feats)

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


best_val_ccc = -float("inf")
os.makedirs("saved_models", exist_ok=True)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0

    random.shuffle(train_idx)

    for i, idx in enumerate(train_idx):
        sample = ds[idx]

        if epoch == 0 and i == 0:
            print("n\ First training sample")

            if MODALITY == "text":
                print("Transcript")
                print(sample["transcription"])

            elif MODALITY == "audio":
                waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))
                print(f"Audio: {len(waveform)} samples @ {sr} Hz")
                print("First 10 waveform values:", waveform[:10])

            print("Target:", [
                sample["EmoVal"],
                sample["EmoAct"],
                sample["EmoDom"]
            ])

        target = torch.tensor(
            [sample["EmoVal"], sample["EmoAct"], sample["EmoDom"]],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        feats = build_features(sample)

        optimizer.zero_grad()
        pred = model(feats)

        diff = torch.abs(target - mu)

        mse = loss_fn(pred, target).mean(dim=1)
        loss = mse.mean()

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

    avg_loss = running_loss / len(train_idx)
    print(f"\nEpoch {epoch + 1} Train Loss: {avg_loss:.4f}")

    val_ccc = evaluate(model, val_idx, "VAL")
    evaluate(model, test_idx, "TEST")

    if val_ccc > best_val_ccc:
        best_val_ccc = val_ccc
        save_path = os.path.join("saved_models", f"best_{MODALITY}_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model (VAL CCC): {save_path}")

print("Training complete.")
print(f"Best validation CCC: {best_val_ccc:.4f}")