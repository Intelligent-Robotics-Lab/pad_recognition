import librosa
import numpy as np
import soundfile as sf

"""
Extract frame-level audio features for emotion modeling
Parameters:
    - audio_path: Path to the audio file
    - target_sr: Target sampling rate for audio processing
    - max_seconds: Maximum duration of audio to process (in seconds)
Returns:
features: np.ndarray of shape (T_audio, 42) where T_audio is the number of time frames and 42 is the feature dimension per frame
    Frame-level audio features including:
    - MFCCs (20)
    - Delta MFCCs (20)
    - Energy (1)
    - Spectral centroid (1)

    T_audio = number of frames determined by hop_length
"""

def extract_audio_features(audio_path, target_sr=16000, max_seconds=None):
    try:
        y, sr = sf.read(audio_path, dtype='float32')
    except Exception:
        return np.zeros((1, 42), dtype=np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if max_seconds is not None:
        y = y[:int(max_seconds * sr)]

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    hop_length = 512
    n_mfcc = 20
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)

    energy = librosa.feature.rms(y=y, hop_length=hop_length)
    energy = np.log(energy + 1e-6)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_centroid = np.log(spectral_centroid + 1e-6)  
    
    T = min(mfcc.shape[1], mfcc_delta.shape[1], energy.shape[1], spectral_centroid.shape[1])

    mfcc = mfcc[:, :T]
    mfcc_delta = mfcc_delta[:, :T]
    energy = energy[:, :T]
    spectral_centroid = spectral_centroid[:, :T]

    # VAD Detection
    energy_vals = energy[0]
    threshold = np.percentile(energy_vals, 20)
    speech_mask = energy_vals > threshold 
    if speech_mask.sum() < 10:
        speech_mask[:] = True

    energy = np.convolve(energy_vals, np.ones(3)/3, mode='same')[None, :]

    stacked_features = np.vstack([
        mfcc, 
        mfcc_delta, 
        energy,
        spectral_centroid
    ]) # shape (42, T)

    stacked_features = stacked_features * speech_mask

    features = stacked_features.T.astype(np.float32) # (T_audio, 42)

    return features