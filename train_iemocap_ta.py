import torch
import soundfile as sf
from datasets import load_dataset, Audio

from io import BytesIO

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features
from models.emotion_model_text_audio import EmotionPADModelTA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading IEMOCAP dataset...")
ds = load_dataset("AbstractTTS/IEMOCAP")["train"]
ds = ds.cast_column("audio", Audio(decode=False))

print(f"Dataset size: {len(ds)}")

model = EmotionPADModelTA(
    text_input_dim=1024,
    audio_input_dim=1024,
).to(device)

model.eval()

num_samples = 5

for i in range(num_samples):
    print(f"SAMPLE {i}")

    sample = ds[i]

    print("File:", sample["file"])
    print("Transcript:", sample["transcription"])
    print("Emotion:", sample["major_emotion"])

    target = torch.tensor(
        [sample["EmoVal"], sample["EmoAct"], sample["EmoDom"]],
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    print("PAD target:", target)

    text_feats = extract_text_features(sample["transcription"])  # [T, 1024]
    text_tensor = text_feats.unsqueeze(0).to(device)

    print("Text tensor:", text_tensor.shape)

    waveform, sr = sf.read(BytesIO(sample["audio"]["bytes"]))

    audio_feats = extract_audio_features(waveform, sr)  # [T, 1024]
    audio_tensor = torch.tensor(audio_feats, dtype=torch.float32).unsqueeze(0).to(device)

    print("Audio tensor:", audio_tensor.shape)

    with torch.no_grad():
        p, a, d = model(text_tensor, audio_tensor)

    pred = torch.cat([p, a, d], dim=1).cpu()

    print("\nPrediction:", pred)
    print("Target    :", target.cpu())
    print("Error     :", (pred.cpu() - target.cpu()))

    print("\nDone.")