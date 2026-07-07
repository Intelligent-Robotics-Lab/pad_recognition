"""
Inference script for single-modality PAD prediction using IEMOCAP.
Matches train_single_modality.py exactly.
"""

import random
from io import BytesIO

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset, Audio

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from models.emotion_model import EmotionPADModel
from models.single_modality_model import SingleModalityModel


MODALITY = "audio"

SEED = 42
USE_GRU = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)

# Plan to update this implementation in future iterations
full_model = EmotionPADModel(
    text_input_dim=1024,
    audio_input_dim=1024,
    video_input_dim=7,
    d_model=512,
    use_gru=USE_GRU,
).to(device)

if MODALITY == "text":
    encoder = full_model.text_encoder
elif MODALITY == "audio":
    encoder = full_model.audio_encoder
else:
    raise ValueError("Only text and audio are currently supported.")

regressor = full_model.pad_regressor

model = SingleModalityModel(
    encoder=encoder,
    pad_regressor=regressor
).to(device)

checkpoint = f"saved_models/best_{MODALITY}_model.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

print(f"Loaded {checkpoint}")

# Dataset loading
print("Loading IEMOCAP...")

ds = load_dataset("AbstractTTS/IEMOCAP")["train"]
ds = ds.cast_column("audio", Audio(decode=False))

indices = list(range(len(ds)))
random.shuffle(indices)

train_split = int(0.8 * len(indices))
val_split = int(0.9 * len(indices))

test_idx = indices[val_split:]

print(f"Testing on {len(test_idx)} samples")

# Same feature extraction as in training
def build_features(sample):

    if MODALITY == "text":
        feats = extract_text_features(sample["transcription"])

    elif MODALITY == "audio":
        waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))
        feats = extract_audio_features(waveform, sr)

    return torch.tensor(feats,dtype=torch.float32).unsqueeze(0).to(device)

# Calculation for all of the metrics to be reporteds
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def pearson_cc(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def ccc(y_true, y_pred):

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)

# Actual inference script
all_preds = []
all_targets = []

with torch.no_grad():
    for idx in test_idx:
        sample = ds[idx]
        target = np.array([
            sample["EmoVal"],
            sample["EmoAct"],
            sample["EmoDom"]
        ])

        feats = build_features(sample)
        pred = model(feats)

        all_preds.append(pred.squeeze(0).cpu().numpy())
        all_targets.append(target)

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# Results reporting

dims = ["Pleasure", "Arousal", "Dominance"]

ccc_scores = []

print(f"{MODALITY.upper()} INFERENCE RESULTS")

for i, dim in enumerate(dims):

    y_true = all_targets[:, i]
    y_pred = all_preds[:, i]

    ccc_val = ccc(y_true, y_pred)
    ccc_scores.append(ccc_val)

    print(f"\n{dim}")
    print(f"CCC      : {ccc_val:.4f}")
    print(f"Pearson  : {pearson_cc(y_true, y_pred):.4f}")
    print(f"RMSE     : {rmse(y_true, y_pred):.4f}")
    print(f"MAE      : {mae(y_true, y_pred):.4f}")

print("\n" + "-" * 55)
print(f"Average CCC : {np.mean(ccc_scores):.4f}")
print("-" * 55)