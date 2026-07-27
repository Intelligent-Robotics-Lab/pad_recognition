"""
Inference script for TEXT + AUDIO modality
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from utils.helpers import PrecomputedDataset, multimodal_collate
from models.emotion_model_text_audio import EmotionPADModelTA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

precomputed_dir = "data/MELD.Raw/precomputed_v4"

chunk_files = sorted([
    os.path.join(precomputed_dir, f)
    for f in os.listdir(precomputed_dir)
    if f.startswith("test_features_")
])

if not chunk_files:
    raise FileNotFoundError(f"No test chunks found in {precomputed_dir}")

datasets = [PrecomputedDataset(f) for f in chunk_files]
test_dataset = ConcatDataset(datasets)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=multimodal_collate
)

model = EmotionPADModelTA(
    text_input_dim=1024, 
    audio_input_dim=1024,
    d_model=512
).to(device)

model.load_state_dict(torch.load("saved_models/best_model_ta.pth", map_location=device))
model.eval()

all_preds = []
all_targets = []

# Inference loop
for batch in test_loader:
    if batch is None:
        continue

    text_feats, audio_feats, _, pad_targets = batch

    text_feats = text_feats.to(device)
    audio_feats = audio_feats.to(device)
    pad_targets = pad_targets.to(device)

    # Optional normalization (match your original script)
    audio_feats = (audio_feats - audio_feats.mean(dim=0)) / (audio_feats.std(dim=0) + 1e-6)

    with torch.no_grad():
        pleasure, arousal, dominance = model(text_feats, audio_feats)
        preds = torch.cat([pleasure, arousal, dominance], dim=1)

    all_preds.append(preds.cpu().numpy())
    all_targets.append(pad_targets.cpu().numpy())

# Stack results
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Metrics (same as your original script)
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def pearson_cc(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

def ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true)*(y_pred - mean_pred))
    return (2*cov) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)

dims = ["Pleasure", "Arousal", "Dominance"]

print("\n=== RESULTS (TEXT + AUDIO) ===")
for i, dim in enumerate(dims):
    y_true = all_targets[:, i]
    y_pred = all_preds[:, i]

    print(f"{dim}:")
    print(f"  RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"  MAE : {mae(y_true, y_pred):.4f}")
    print(f"  Pearson CC: {pearson_cc(y_true, y_pred):.4f}")
    print(f"  CCC : {ccc(y_true, y_pred):.4f}")
    print("-"*40)