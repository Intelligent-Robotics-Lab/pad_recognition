import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, Audio
from io import BytesIO
import soundfile as sf

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from models.emotion_model_text_audio import EmotionPADModelTA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
learning_rate = 5e-4
seed = 42

random.seed(seed)
torch.manual_seed(seed)

model = EmotionPADModelTA(
    text_input_dim=1024,
    audio_input_dim=1024,
    d_model=512
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# CCCs for evaluation only
@torch.no_grad()
def ccc_score(pred, target):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)

    pred_var = pred.var(dim=0)
    target_var = target.var(dim=0)

    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)

    ccc = (2 * cov) / (
        pred_var + target_var +
        (pred_mean - target_mean).pow(2) + 1e-8
    )
    return ccc.mean().item()

print("Loading IEMOCAP...")
ds = load_dataset("AbstractTTS/IEMOCAP")["train"]
ds = ds.cast_column("audio", Audio(decode=False))

print("Dataset size:", len(ds))

# Split the dataset
indices = list(range(len(ds)))
random.shuffle(indices)

train_split = int(0.8 * len(indices))
val_split   = int(0.9 * len(indices))

train_idx = indices[:train_split]
val_idx   = indices[train_split:val_split]
test_idx  = indices[val_split:]

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
                device=device
            ).unsqueeze(0)

            text = extract_text_features(sample["transcription"]).unsqueeze(0).to(device)

            waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))
            audio = extract_audio_features(waveform, sr)
            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

            p, a, d = model(text, audio)
            pred = torch.cat([p, a, d], dim=1)

            preds_all.append(pred)
            targets_all.append(target)

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    score = ccc_score(preds_all, targets_all)

    print(f"{name} CCC: {score:.4f}")

    model.train()
    return score

# Training loop
best_val_ccc = -float("inf")

os.makedirs("saved_models", exist_ok=True)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    running_loss = 0.0

    random.shuffle(train_idx)

    for i, idx in enumerate(train_idx):

        sample = ds[idx]

        target = torch.tensor(
            [sample["EmoVal"], sample["EmoAct"], sample["EmoDom"]],
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        text_feats = extract_text_features(sample["transcription"])
        text_feats = text_feats.unsqueeze(0).to(device)

        waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))
        audio_feats = extract_audio_features(waveform, sr)
        audio_feats = torch.tensor(audio_feats, dtype=torch.float32).unsqueeze(0).to(device)

        optimizer.zero_grad()

        p, a, d = model(text_feats, audio_feats)
        pred = torch.cat([p, a, d], dim=1)

        loss = loss_fn(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

        if i % 200 == 0:
            print(f"\nSample {i}")
            print("Pred:", pred.detach().cpu())
            print("Target:", target.detach().cpu())
            print("Loss:", loss.item())

    avg_loss = running_loss / len(train_idx)
    print(f"\nEpoch {epoch+1} Train MSE: {avg_loss:.4f}")

    # Validation with CCC
    val_ccc = evaluate(model, val_idx, "VAL")

    # Optional test evaluation
    test_ccc = evaluate(model, test_idx, "TEST")

    # Save the best model only
    if val_ccc > best_val_ccc:
        best_val_ccc = val_ccc
        torch.save(model.state_dict(), "saved_models/iemocap_ta_best.pth")
        print("Saved new best model (based on VAL CCC)")