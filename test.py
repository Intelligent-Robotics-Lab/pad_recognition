import torch
import pandas as pd

from features.audio_features import extract_audio_features
from features.video_features import extract_video_features
from features.text_features import extract_text_features
from models.emotion_model import EmotionPADModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model with correct input dimensions
model = EmotionPADModel(
    text_input_dim=768,
    audio_input_dim=63,
    video_input_dim=52,
    d_model=512
).to(device)

model.eval()

# Paths to one MELD sample
audio_path = "/home/carter/pad_recognition/data/MELD.Raw/train/dia0_utt0.wav"
video_path = "/home/carter/pad_recognition/data/MELD.Raw/train/dia0_utt0.mp4"
meld_csv  = "/home/carter/pad_recognition/data/MELD.Raw/train_sent_emo.csv"

# 1. TEXT FEATURES
df = pd.read_csv(meld_csv)
first_text = df.iloc[0]["Utterance"]

text_features = extract_text_features(first_text)        # [T_text, 768]
text_features = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0).to(device)

# 2. AUDIO FEATURES
audio_features = extract_audio_features(audio_path)      # [T_audio, 63]
audio_features = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device)

# 3. VIDEO FEATURES
video_features = extract_video_features(
    video_path,
    sample_fps=5,
    max_seconds=None,
    resize_dim=(256, 256),
    include_deltas=True
)                                                       # [T_video, 52]
video_features = torch.tensor(video_features, dtype=torch.float32).unsqueeze(0).to(device)

# PRINT SHAPES + SAMPLE VALUES
print("\n===== FEATURE SHAPES =====")
print(f"Text features:  {text_features.shape}   (expected: [1, T_text, 768])")
print(f"Audio features: {audio_features.shape}  (expected: [1, T_audio, 63])")
print(f"Video features: {video_features.shape}  (expected: [1, T_video, 52])")

# Forward pass
with torch.no_grad():
    p,a,d = model(text_features, audio_features,video_features)

print("\n===== MODEL OUTPUT (RANDOM WEIGHTS) =====")
print(f"P = {p.item():.4f}")
print(f"A = {a.item():.4f}")
print(f"D = {d.item():.4f}")
