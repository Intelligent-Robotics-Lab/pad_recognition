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
features: np.ndarray of shape (T_audio, 63) where T_audio is the number of time frames and 63 is the feature dimension per frame
    Frame-level audio features including:
    - MFCCs (20)
    - Delta MFCCs (20)
    - Delta-Delta MFCCs (20)
    - Energy (1)
    - Pitch (1)
    - Spectral centroid (1)

    T_audio = number of frames determined by hop_length
"""

def extract_audio_features(audio_path, target_sr=16000, max_seconds=None):
    # Load in the audio file
    y, sr = sf.read(audio_path, dtype='float32')

    # Convert stereo to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Trim to max seconds if provided
    if max_seconds is not None:
        y = y[:int(max_seconds * sr)]
    
    # Resample if necessary
    if sr != target_sr:
        y = librosa.resample(y.T, orig_sr=sr, target_sr=target_sr).T
        sr = target_sr

    # Feature extraction with reduced frames
    hop_length = 512
    n_mfcc = 20
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Obtain energy, pitch, spectral centroid calculations from the audio signal
    energy = librosa.feature.rms(y=y, hop_length=hop_length)
    pitch = librosa.yin(y=y, fmin=50, fmax=300, frame_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)

    # **Improvement could come from better handling of truncation and non-speech frames, as well as more robust normalization.**

    # Align all features to same number of frames
    T = min(mfcc.shape[1], mfcc_delta.shape[1], mfcc_delta2.shape[1], energy.shape[1], pitch.shape[0], spectral_centroid.shape[1])

    mfcc = mfcc[:, :T]
    mfcc_delta = mfcc_delta[:, :T]
    mfcc_delta2 = mfcc_delta2[:, :T]
    energy = energy[:, :T]
    pitch = pitch[:T]
    spectral_centroid = spectral_centroid[:, :T]

    # Voice activity detection (VAD) to filter out non-speech frames
    speech_mask = energy[0] > 0.01  # Simple energy threshold for VAD
    if np.sum(speech_mask) < 1: # prevent empty mask
        speech_mask[:] = True

    # Interpolation for non-speech frames
    def interpolate_features(f, mask):
        "Interpolate missing frames linearly to prevent massive drops to 0"
        indices = np.arange(len(f))
        valid = indices[mask]
        return np.interp(indices, valid, f[mask])

    # Apply the interpolation to all 63 features
    stacked_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2, energy, pitch[np.newaxis, :], spectral_centroid])
    for i in range(stacked_features.shape[0]):
        stacked_features[i] = interpolate_features(stacked_features[i], speech_mask)

    # Normalize using speech frames
    mean = stacked_features[:,speech_mask].mean(axis=1, keepdims=True)
    std = stacked_features[:,speech_mask].std(axis=1, keepdims=True) + 1e-6 # avoid the divide by zero
    stacked_features = (stacked_features - mean) / std

    # Transpose to (T_audio, feature_dim)
    features = stacked_features.T.astype(np.float32) # (T_audio, 63)

    return features