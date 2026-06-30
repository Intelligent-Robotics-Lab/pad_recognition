import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import soundfile as sf
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(device)
model.eval()

def extract_wav2vec_features(audio_path, target_sr=16000, max_seconds=None):
    try:
        y, sr = sf.read(audio_path, dtype='float32')
    except Exception as e:
        print(f"Audio load failed: {audio_path} -> {e}")
        return np.zeros((1, 1024), dtype=np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if max_seconds is not None:
        y = y[:int(max_seconds * sr)]

    if sr != target_sr:
        y = torchaudio.functional.resample(
            torch.tensor(y), sr, target_sr
        ).numpy()

    y = (y - np.mean(y)) / (np.std(y) + 1e-7)

    inputs = processor(y, sampling_rate=target_sr, return_tensors="pt")

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    return outputs.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)