"""
Inference script for SINGLE modality (text / audio / video)
Matches train_single_modality.py behavior exactly
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from utils.helpers import PrecomputedDataset, multimodal_collate
from models.emotion_model import EmotionPADModel
from models.single_modality_model import SingleModalityModel

# Change this line for different modalities
MODALITY = "text"   # "text", "audio", "video"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

precomputed_dir = "data/MELD.Raw/precomputed_v4"

# Load dataset
chunk_files = sorted([
    os.path.join(precomputed_dir, f)
    for f in os.listdir(precomputed_dir)
    if f.startswith("test_features_")
])

datasets = [PrecomputedDataset(f) for f in chunk_files]
test_dataset = ConcatDataset(datasets)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=multimodal_collate
)

full_model = EmotionPADModel(
    text_input_dim=1024,
    audio_input_dim=1024,
    video_input_dim=7,
    d_model=512
).to(device)

# Select encoders
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

model.load_state_dict(torch.load(f"saved_models/best_{MODALITY}_model.pth", map_location=device))
model.eval()

all_preds = []
all_targets = []

def get_modality_batch(batch, modality):
    text, audio, video, target = batch

    if modality == "text":
        x = text
    elif modality == "audio":
        x = audio
    else:
        x = video

    return x, target

for batch in test_loader:
    if batch is None:
        continue

    x, targets = get_modality_batch(batch, MODALITY)

    x = x.to(device)
    targets = targets.to(device)

    # # SAME as training // removed due to errors
    # if x.dim() == 3:
    #     x = x.mean(dim=1)

    with torch.no_grad():
        preds = model(x)

    all_preds.append(preds.cpu().numpy())
    all_targets.append(targets.cpu().numpy())

# Stack results
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Metrics (same as main infer script)
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

print(f"\nRESULTS ({MODALITY.upper()})")
for i, dim in enumerate(dims):
    y_true = all_targets[:, i]
    y_pred = all_preds[:, i]

    print(f"{dim}:")
    print(f"  RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"  MAE : {mae(y_true, y_pred):.4f}")
    print(f"  Pearson CC: {pearson_cc(y_true, y_pred):.4f}")
    print(f"  CCC : {ccc(y_true, y_pred):.4f}")
    print("-"*40)