from io import BytesIO

import soundfile as sf
import torch
from datasets import load_dataset, Audio

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from models.emotion_model_text_audio import EmotionPADModelTA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")

ds = load_dataset("AbstractTTS/IEMOCAP")["train"]
ds = ds.cast_column("audio", Audio(decode=False))

sample_idx = 5
sample = ds[sample_idx]

print(f"Testing sample {sample_idx}")
print("Loaded sample:", sample["file"])
print()


print("GROUND TRUTH")

print("Transcript :", sample["transcription"])
print("Emotion    :", sample["major_emotion"])

target = torch.tensor(
    [
        sample["EmoVal"],
        sample["EmoAct"],
        sample["EmoDom"],
    ],
    dtype=torch.float32,
).unsqueeze(0)

print("PAD:", target)
print()


print("TEXT")

text_features = extract_text_features(sample["transcription"])

print("Raw text features:", text_features.shape)

text_tensor = text_features.unsqueeze(0).to(device)

print("Text tensor:", text_tensor.shape)
print()

print("AUDIO")

waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))

print("Waveform shape:", waveform.shape)
print("Sample rate   :", sr)

audio_features = extract_audio_features(
    waveform,
    sr,
)

print("Raw audio features:", audio_features.shape)

audio_tensor = audio_features.unsqueeze(0).to(device)

print("Audio tensor:", audio_tensor.shape)
print()


print("MODEL")

model = EmotionPADModelTA(
    text_input_dim=1024,
    audio_input_dim=1024,
).to(device)

model.eval()

with torch.no_grad():
    p, a, d = model(
        text_tensor,
        audio_tensor,
    )

prediction = torch.cat([p, a, d], dim=1).cpu()


print()
print("RESULT")

print("Prediction shape:", prediction.shape)
print(prediction)

print()

print("Ground Truth:")
print(target)

print()

print("Difference:")
print(prediction - target)