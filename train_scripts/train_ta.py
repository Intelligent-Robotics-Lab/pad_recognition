"""
Train script to test the audio and video modalaties together using the IEMOCAP dataset.
"""

import os
import random
from io import BytesIO

import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, Audio

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

# Load pretrained weights from unimodal encoders (train_ind.py file)
text_checkpoint = torch.load("saved_models/best_text_model.pth", map_location=device)

audio_checkpoint = torch.load("saved_models/best_audio_model.pth", map_location=device)

# Will update the logic in train_ind.py 
text_encoder_state = {
    k.replace("encoder.", ""): v
    for k, v in text_checkpoint.items()
    if k.startswith("encoder.")
}

audio_encoder_state = {
    k.replace("encoder.", ""): v
    for k, v in audio_checkpoint.items()
    if k.startswith("encoder.")
}

model.text_encoder.load_state_dict(text_encoder_state)
model.audio_encoder.load_state_dict(audio_encoder_state)

print("Loaded pretrained encoders.")

# Immediately freeze the encoders so their weights are saved
for param in model.text_encoder.parameters():
    param.requires_grad = False

for param in model.audio_encoder.parameters():
    param.requires_grad = False

# Now only the fusion layer and PAD regressor will update
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
loss_fn = nn.SmoothL1Loss(reduction="none")

# Verify the frozen paramters
print("\nTrainable Paramters:")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

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

print("Loading IEMOCAP...")
ds = load_dataset("AbstractTTS/IEMOCAP")["train"]
ds = ds.cast_column("audio", Audio(decode=False))

print("Dataset size:", len(ds))

# Compute the dataset mean to use in weighted loss function
targets = torch.tensor(
    [[ds[i]["EmoVal"], ds[i]["EmoAct"], ds[i]["EmoDom"]] for i in range(len(ds))],
    dtype = torch.float32
)

mu = targets.mean(dim=0).to(device) # shape (3,)

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

            text_feats = extract_text_features(sample["transcription"]).unsqueeze(0).to(device)

            waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))
            audio_feats = extract_audio_features(waveform, sr)
            audio_feats = torch.tensor(audio_feats, dtype=torch.float32).unsqueeze(0).to(device)

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

# Training loop
best_val_ccc = -float("inf")
os.makedirs("saved_models", exist_ok=True)

# Early stopping variables
patience = 3
epochs_without_improvement = 0

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

        text_feats = extract_text_features(sample["transcription"]).unsqueeze(0).to(device)

        waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))
        audio_feats = extract_audio_features(waveform, sr)
        audio_feats = torch.tensor(audio_feats, dtype=torch.float32).unsqueeze(0).to(device)

        optimizer.zero_grad()
        pred = model(text_feats, audio_feats)

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
    print(f"\nEpoch {epoch+1} Train MSE: {avg_loss:.4f}")

    # Validation with CCC
    val_ccc = evaluate(model, val_idx, "VAL")

    min_delta = 1e-3

    # Save the best model only
    if val_ccc > best_val_ccc + min_delta:
        best_val_ccc = val_ccc
        epochs_without_improvement = 0

        save_path = os.path.join("saved_models", f"best_ta_model.pth")
        torch.save(model.state_dict(), save_path)

        print(
            f"Saved new best model "
            f"(VAL CCC = {best_val_ccc:.4f})"
        )
    
    else:
        epochs_without_improvement += 1

        print(f"No improvements for {epochs_without_improvement}/{patience} epochs.")

        if epochs_without_improvement >= patience:
            print("\nEarly stopping triggered.")
            break

print("Training complete.")
print(f"Best validation CCC: {best_val_ccc:.4f}")