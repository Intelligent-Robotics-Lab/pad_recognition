"""
Inference script for the Text + Audio PAD model. Matches train_ta.py behavior exactly.
"""

import os
from io import BytesIO
import random

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from models.emotion_model_text_audio import EmotionPADModelTA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

model = EmotionPADModelTA(
    text_input_dim=1024,
    audio_input_dim=1024,
    d_model=512
).to(device)

checkpoint = "saved_models/best_ta_model.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

print(f"Loaded {checkpoint}")

print("Loading IEMOCAP...")
ds = load_dataset("AbstractTTS/IEMOCAP")["train"]
ds = ds.cast_column("audio", Audio(decode=False))

indices = list(range(len(ds)))
random.shuffle(indices)

train_split = int(0.8 * len(indices))
val_split = int(0.9 * len(indices))

test_idx = indices[val_split:]

print(f"Testing on {len(test_idx)} samples")

# Calculated metrics
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def pearson(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    return (2 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8))

# Inference script
predictions = []
targets = []

with torch.no_grad():

    for idx in test_idx:

        sample = ds[idx]

        target = np.array([
            sample["EmoVal"],
            sample["EmoAct"],
            sample["EmoDom"]
        ])

        # Normalize to the [-1, 1] range
        target = (target - 3.0) / 2.0

        # Text
        text_feats = extract_text_features(sample["transcription"]).unsqueeze(0).to(device)

        # Audio
        waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))

        audio_feats = extract_audio_features(waveform, sr)
        audio_feats = torch.tensor(audio_feats, dtype=torch.float32).unsqueeze(0).to(device)

        pred = model(text_feats, audio_feats)

        predictions.append(pred.squeeze(0).cpu().numpy())
        targets.append(target)

predictions = np.array(predictions)
targets = np.array(targets)

# Printed results section
dims = ["Pleasure", "Arousal", "Dominance"]
ccc_scores = []

print("\nTEXT + AUDIO INFERENCE RESULTS\n")

for i, dim in enumerate(dims):

    y_true = targets[:, i]
    y_pred = predictions[:, i]

    score = ccc(y_true, y_pred)
    ccc_scores.append(score)

    print(dim)
    print(f"CCC      : {score:.4f}")
    print(f"Pearson  : {pearson(y_true, y_pred):.4f}")
    print(f"RMSE     : {rmse(y_true, y_pred):.4f}")
    print(f"MAE      : {mae(y_true, y_pred):.4f}")
    print()

print(f"Average CCC : {np.mean(ccc_scores):.4f}")
