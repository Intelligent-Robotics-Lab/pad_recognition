"""
Inference script for the full text+audio+video PAD model on MELD's test split.
Matches train.py exactly (on-the-fly feature extraction) and mirrors
inference_scripts/IEMOCAP/infer_multimodal.py's structure, adapted for MELD's single
fixed test split instead of LOSO folds — see docs/decision_log.md 2026-08-03.

(inference_scripts/IEMOCAP/infer_multimodal.py itself currently has real bugs — wrong
model import/class name and a syntax error — so this mirrors infer_ind.py's working
pattern instead of copying that file's broken code.)
"""

import os
import numpy as np
import torch

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features

from models.emotion_model import EmotionPADModel

from utils.dataloaders import get_meld_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Valid inputs "mlp" and "transformer" — must match what train.py actually trained.
FUSION_TYPE = "transformer"

SEED = 42
torch.manual_seed(SEED)

model = EmotionPADModel(
    text_hidden_dim=1024,
    audio_input_dim=1024,
    video_input_dim=1280,
    d_model=512,
    fusion_type=FUSION_TYPE
).to(device)

# Evaluation metrics
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

checkpoint = f"saved_models/best_meld_tav_{FUSION_TYPE}.pth"

if not os.path.exists(checkpoint):
    raise FileNotFoundError(f"No checkpoint found at {checkpoint} — run train_multimodal_meld.py first.")

model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

print(f"\nLoaded {checkpoint}")

_, _, test_loader = get_meld_loaders(batch_size=1)

predictions = []
targets = []

with torch.no_grad():
    for batch in test_loader:
        text = batch["text"][0]
        audio = batch["audio"][0]
        sample_rate = batch["sample_rate"][0]
        video_path = batch["video_path"][0]
        start_time = batch["start_time"][0]
        end_time = batch["end_time"][0]

        target = batch["pad"].squeeze(0).cpu().numpy()

        text_feats = extract_text_features(text)
        audio_feats = extract_audio_features(audio, sample_rate)
        video_feats = extract_video_features(video_path, start_time, end_time)

        text_feats = torch.tensor(text_feats, dtype=torch.float32, device=device)
        audio_feats = torch.tensor(audio_feats, dtype=torch.float32, device=device)
        video_feats = torch.tensor(video_feats, dtype=torch.float32, device=device)

        if text_feats.dim() == 2:
            text_feats = text_feats.unsqueeze(0)
        if audio_feats.dim() == 2:
            audio_feats = audio_feats.unsqueeze(0)
        if video_feats.dim() == 2:
            video_feats = video_feats.unsqueeze(0)

        pred = model(text_feats, audio_feats, video_feats)

        predictions.append(pred.squeeze(0).cpu().numpy())
        targets.append(target)

predictions = np.array(predictions)
targets = np.array(targets)

# Results reporting
dims = ["Pleasure", "Arousal", "Dominance"]
ccc_scores = []

for i, dim in enumerate(dims):
    y_true = targets[:, i]
    y_pred = predictions[:, i]

    score = ccc(y_true, y_pred)
    ccc_scores.append(score)

    print(f"\n{dim}")
    print(f"CCC      : {score:.4f}")
    print(f"Pearson  : {pearson(y_true, y_pred):.4f}")
    print(f"RMSE     : {rmse(y_true, y_pred):.4f}")
    print(f"MAE      : {mae(y_true, y_pred):.4f}")

print(f"\nAverage CCC: {np.mean(ccc_scores):.4f}")
