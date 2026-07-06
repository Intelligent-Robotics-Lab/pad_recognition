import torch
import soundfile as sf
from io import BytesIO
from datasets import load_dataset, Audio

from features.text_features import extract_text_features
from models.encoders import TextTransformerEncoder

# Load without auto audio decoding
ds = load_dataset("AbstractTTS/IEMOCAP")["train"]
ds = ds.cast_column("audio", Audio(decode=False))

sample = ds[0]

print("Sample Information")

print(f"File: {sample['file']}")
print(f"Transcript: {sample['transcription']}")
print(f"Major Emotion: {sample['major_emotion']}")
print()

print("PAD Labels")
print(f"  Valence   : {sample['EmoVal']}")
print(f"  Arousal   : {sample['EmoAct']}")
print(f"  Dominance : {sample['EmoDom']}")

print("\nEncoding text...")

text = sample["transcription"]

text_features = extract_text_features(text)      # [T, 1024]
text_tensor = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0)
encoder = TextTransformerEncoder(hidden_dim=1024, d_model=512)
encoded_text = encoder(text_tensor)

print("Raw text feature shape :", text_tensor.shape)
print("Encoded text shape     :", encoded_text.shape)


print("\nLoading audio from stored bytes...")

audio_bytes = sample["audio"]["bytes"]

waveform, sr = sf.read(BytesIO(audio_bytes))

print("Waveform shape :", waveform.shape)
print("Sampling rate  :", sr)
print("Duration (sec) :", len(waveform) / sr)


pad_target = torch.tensor([sample["EmoVal"], sample["EmoAct"], sample["EmoDom"],], dtype=torch.float32,)

print("\nPAD Target Tensor")
print(pad_target)
print("Target shape:", pad_target.shape)