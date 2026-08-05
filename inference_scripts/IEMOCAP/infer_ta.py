"""
Inference script for the Text + Audio PAD model. Matches train_ta.py behavior exactly.
"""

import os
import numpy as np
import torch

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from models.emotion_model_text_audio import EmotionPADModelTA

from utils.dataloaders import get_iemocap_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Valid inputs "mlp" and "transformer"
FUSION_TYPE = "transformer"

SEED = 42

torch.manual_seed(SEED)

# Build the same text-audio model used during training
model = EmotionPADModelTA(text_input_dim=1024, audio_input_dim=1024, d_model=512, fusion_type=FUSION_TYPE).to(device)

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

def evaluate_fold(fold):

    checkpoint = f"saved_models/best_ta_{FUSION_TYPE}_loso_fold{fold}.pth"

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
        fold=fold
    )

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in test_loader:

            # Read a sample from the dataloader
            text = batch["text"][0]
            audio = batch["audio"][0]
            sample_rate = batch["sample_rate"][0]

            target = batch["pad"].squeeze(0).cpu().numpy()

            text_feats = extract_text_features(text)
            audio_feats = extract_audio_features(audio, sample_rate)

            text_feats = torch.tensor(text_feats, dtype=torch.float32, device=device,)
            audio_feats = torch.tensor(audio_feats, dtype=torch.float32, device=device,)

            # Match training script dimensions
            if text_feats.dim() == 2:
                text_feats = text_feats.unsqueeze(0)

            if audio_feats.dim() == 2:
                audio_feats = audio_feats.unsqueeze(0)

            pred = model(text_feats, audio_feats)

            predictions.append(pred.squeeze(0).cpu().numpy())
            targets.append(target)

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
        print(f"Pearson  : {pearson(y_true, y_pred):.4f}")
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
    print(F"Fold {i}: {score:.4f}")

print(f"\nPleasure CCC: {np.mean(pleasure_results):.4f}")
print(f"\nArousal CCC: {np.mean(arousal_results):.4f}")
print(f"\nDominance CCC: {np.mean(dominance_results):.4f}")
print(f"Average CCC : {np.mean(fold_results):.4f}")