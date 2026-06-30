"""
Test script for debugging the feature extraction and model inference pipeline on a single MELD sample.
This ensures all modalities (text, audio, video) work correctly before full training.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
torch.backends.cudnn.enabled = False
import pandas as pd

from features.text_features import extract_text_features
# from features.video_features_v3 import extract_pyfeat_features
from features.video_features_v2 import extract_emotion_probs
from features.audio_features_v2 import extract_wav2vec_features
from models.emotion_model import EmotionPADModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_path = "/home/carter/pad_recognition/data/MELD.Raw/train/dia0_utt0.wav"
video_path = "/home/carter/pad_recognition/data/MELD.Raw/train/dia0_utt0.mp4"
meld_csv  = "/home/carter/pad_recognition/data/MELD.Raw/train_sent_emo.csv"

df = pd.read_csv(meld_csv)
first_text = df.iloc[0]["Utterance"]

text_features = extract_text_features(first_text)  # [T, 1024]
text_features = torch.as_tensor(text_features, dtype=torch.float32).unsqueeze(0).to(device)

audio_features = extract_wav2vec_features(audio_path)  # [T, 1024]
audio_features = torch.as_tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)

video_features = extract_emotion_probs(video_path)  # [T, 7]
video_features = torch.as_tensor(video_features, dtype=torch.float32).unsqueeze(0).to(device)

# video_tensor, video_meta = extract_pyfeat_features(video_path, sample_fps=5, include_deltas=True)
# video_features = video_tensor.unsqueeze(0)
# video_dim = video_features.shape[-1]

# print(f"[INFO] Video valid_ratio: {video_meta['valid_ratio']:.2%}")
# if video_meta['valid_ratio'] < 0.5:
#     print("  ⚠ WARNING: fewer than 50% of frames had a detected face")

model = EmotionPADModel(
    text_input_dim=1024,
    audio_input_dim=1024,
    video_input_dim=7,
    d_model=512
).to(device)

model.eval()

print("\n===== FEATURE SHAPES =====")
print(f"Text:  {text_features.shape}")
print(f"Audio: {audio_features.shape}")
print(f"Video: {video_features.shape}")

print("Audio std:", audio_features.std().item())
print("Video std:", video_features.std().item())

# TEST 1 — NUMERIC HEALTH CHECK (NaN / INF)
print("\n===== NUMERIC CHECK =====")

print("NaN - Text/Audio/Video:",
      torch.isnan(text_features).any().item(),
      torch.isnan(audio_features).any().item(),
      torch.isnan(video_features).any().item())

print("INF - Text/Audio/Video:",
      torch.isinf(text_features).any().item(),
      torch.isinf(audio_features).any().item(),
      torch.isinf(video_features).any().item())

# TEST 2 — FEATURE HEALTH (RANGE + DYNAMICS)
print("\n===== FEATURE HEALTH =====")

print("Audio min/max:", audio_features.min().item(), audio_features.max().item())
print("Audio temporal std:", audio_features.std(dim=1).mean().item())

print("Text min/max:", text_features.min().item(), text_features.max().item())
print("Video min/max:", video_features.min().item(), video_features.max().item())

# TEST 3 — MODEL STABILITY
print("\n===== MODEL STABILITY =====")

with torch.no_grad():
    outputs = [
        model(text_features, audio_features, video_features)
        for _ in range(3)
    ]

for i, (p, a, d) in enumerate(outputs):
    print(f"Run {i+1}: P={p.item():.4f}, A={a.item():.4f}, D={d.item():.4f}")