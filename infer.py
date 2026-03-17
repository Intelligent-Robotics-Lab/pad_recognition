"""
Evaluation script for the EmotionPADModel using precomputed MELD features.

Pipeline:
    - Load precomputed test features (text, audio, video)
    - Batch using multimodal_collate
    - Forward pass through the trained model
    - Collect predictions and compute regression metrics:
        RMSE, MAE, Pearson Correlation, CCC
"""

import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from utils.helpers import PrecomputedDataset, multimodal_collate
from models.emotion_model import EmotionPADModel
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load precomputed test chunks
precomputed_dir = "data/MELD.Raw/precomputed"

chunk_files = sorted([
    os.path.join(precomputed_dir, f)
    for f in os.listdir(precomputed_dir)
    if f.startswith("test_features_")
])

if not chunk_files:
    raise FileNotFoundError(f"No precomputed test chunks found in {precomputed_dir}")

# Load all chunks
datasets = [PrecomputedDataset(f) for f in chunk_files]
test_dataset = ConcatDataset(datasets)
test_loader = DataLoader(
    test_dataset, 
    batch_size=8, 
    shuffle=False, 
    collate_fn=multimodal_collate
)

# Load the trained model
model = EmotionPADModel(
    text_input_dim=768, 
    audio_input_dim=63, 
    video_input_dim=21, 
    d_model=512
).to(device)

model.load_state_dict(torch.load("saved_models/emotionpad_trained.pth", map_location=device))
model.eval()

# Metrics accumulators
all_preds = []
all_targets = []

# Iterate over test batches
for batch_idx, (text_feats, audio_feats, video_feats, pad_targets) in enumerate(test_loader):
    text_feats = text_feats.to(device)
    audio_feats = audio_feats.to(device)
    video_feats = video_feats.to(device)
    pad_targets = pad_targets.to(device)

    # Optionally normalize the data (better done during precompute)
    audio_feats = (audio_feats - audio_feats.mean(dim=0)) / (audio_feats.std(dim=0) + 1e-6)
    video_feats = (video_feats - video_feats.mean(dim=0)) / (video_feats.std(dim=0) + 1e-6)

    with torch.no_grad():
        pleasure, arousal, dominance = model(text_feats, audio_feats, video_feats)
        preds = torch.cat([pleasure, arousal, dominance], dim=1)  # [batch, 3]

    all_preds.append(preds.cpu().numpy())
    all_targets.append(pad_targets.cpu().numpy())

# Convert to numpy arrays
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Matric calculations
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def pearson_cc(y_true, y_pred):
    """Pearson Correlation Coefficient"""
    return np.corrcoef(y_true, y_pred)[0,1]

def ccc(y_true, y_pred):
    """Concordance Correlation Coefficient"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true)*(y_pred - mean_pred))
    return (2*cov) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)

# Compute metrics per PAD dimension
dims = ["Pleasure", "Arousal", "Dominance"]
for i, dim in enumerate(dims):
    y_true = all_targets[:, i]
    y_pred = all_preds[:, i]
    print(f"{dim}:")
    print(f"  RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"  MAE : {mae(y_true, y_pred):.4f}")
    print(f"  Pearson CC: {pearson_cc(y_true, y_pred):.4f}")
    print(f"  CCC : {ccc(y_true, y_pred):.4f}")
    print("-"*40)