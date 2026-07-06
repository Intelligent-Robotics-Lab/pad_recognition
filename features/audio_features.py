import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2Processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)

model.eval()

def extract_audio_features(waveform, sampling_rate, target_sr=16000, max_seconds=None,):

    # Convert to tensor
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    else:
        waveform = waveform.float()

    # Stereo to mono if necessary
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)

    # Optional truncation
    if max_seconds is not None:
        waveform = waveform[: int(max_seconds * sampling_rate)]

    # Resample if needed
    if sampling_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sampling_rate, target_sr,)

    # Per-sample normalization
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)

    # Processor
    inputs = processor(waveform.numpy(), sampling_rate=target_sr, return_tensors="pt",)
    input_values = inputs.input_values.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_values)

    # [T, hidden_size]
    return outputs.last_hidden_state.squeeze(0).cpu()

def prepare_audio_features(waveform, sampling_rate):
    return extract_audio_features(waveform, sampling_rate)