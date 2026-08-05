"""
Inference script for single-modality PAD prediction using IEMOCAP. Matches train_ind.py exactly.
"""
import os

import numpy as np
import torch

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features

from models.encoders import (TextTransformerEncoder, AudioProjectionEncoder, VideoProjectionEncoder)
from models.pad_regressor import PADRegressors

from models.single_modality_model import SingleModalityModel

from utils.dataloaders import get_iemocap_loaders

MODALITY = "video"

SEED = 42
USE_GRU = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)

if MODALITY == "text":
    encoder = TextTransformerEncoder(hidden_dim=1024, d_model=512,)

elif MODALITY == "audio":
    encoder = AudioProjectionEncoder(input_dim=1024, d_model=512,)

elif MODALITY == "video":
    encoder = VideoProjectionEncoder(input_dim=1280, d_model=512)

else:
    raise ValueError("Only text, audio, and video are currently supported.")

regressor = PADRegressors(d_model=512, hidden_dim=256,)

model = SingleModalityModel(encoder=encoder, pad_regressor=regressor).to(device)

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
def evaluate_fold(fold):

    checkpoint = f"saved_models/best_{MODALITY}_loso_fold{fold}.pth"

    if not os.path.exists(checkpoint):
        print(f"Skipping Fold {fold}")
        return None, None

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    print(f"\nLoaded {checkpoint}")

    _,_, test_loader = get_iemocap_loaders(
        "data/iemocap.csv",
        batch_size=1,
        split="loso",
        fold=fold,
    )

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in test_loader:

            text = batch["text"][0]
            audio = batch["audio"][0]
            video_path = batch["video_path"][0]
            start_time = batch["start_time"][0]
            end_time = batch["end_time"][0]

            target = batch["pad"].to(device)

            if MODALITY == "text":
                feats = extract_text_features(text)
            
            elif MODALITY == "audio":
                sample_rate = batch["sample_rate"][0]
                feats = extract_audio_features(audio, sample_rate)

            elif MODALITY == "video":
                feats = extract_video_features(video_path, start_time, end_time)
            
            feats = torch.tensor(feats, dtype=torch.float32, device=device)

            if feats.dim() == 2:
                feats = feats.unsqueeze(0)

            pred = model(feats)

            predictions.append(pred.squeeze(0).cpu().numpy())
            targets.append(target.squeeze(0).cpu().numpy())

    return np.array(predictions), np.array(targets)

# Results reporting
dims = ["Pleasure", "Arousal", "Dominance"]
fold_results = []

pleasure_results = []
arousal_results = []
dominance_results = []

for fold in range(1,6):

    predictions, targets = evaluate_fold(fold)

    if predictions is None:
        continue 

    ccc_scores = []

    print(f"\nFold {fold}")

    for i, dim in enumerate(dims):

        y_true = targets[:, i]
        y_pred = predictions[:, i]

        score = ccc(y_true, y_pred)

        ccc_scores.append(score)

        print(f"\n{dim}")
        print(f"CCC      : {score:.4f}")
        print(f"Pearson  : {pearson_cc(y_true, y_pred):.4f}")
        print(f"RMSE     : {rmse(y_true, y_pred):.4f}")
        print(f"MAE      : {mae(y_true, y_pred):.4f}")

    pleasure_results.append(ccc_scores[0])
    arousal_results.append(ccc_scores[1])
    dominance_results.append(ccc_scores[2])

    avg_ccc = np.mean(ccc_scores)
    fold_results.append(avg_ccc)

    print(f"\nFold {fold} Average CCC: {avg_ccc:.4f}")

print("Final LOSO Results")

for i, score in enumerate(fold_results, start=1):
    print(f"Fold {i}: {score:.4f}")

print(f"\nPleasure CCC: {np.mean(pleasure_results):.4f}")
print(f"\nArousal CCC: {np.mean(arousal_results):.4f}")
print(f"\nDominance CCC: {np.mean(dominance_results):.4f}")
print(f"\nAverage CCC: {np.mean(fold_results):.4f}")