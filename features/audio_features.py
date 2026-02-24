import librosa
import numpy as np
import soundfile as sf

def extract_audio_features(audio_path, target_sr=16000, max_seconds=5):
    # Load in the audio file to extract features from
    # Removed some downsampling features throughout to prioritize accuracy over latency
    y, sr = sf.read(audio_path, dtype='float32')

    # Convert stereo to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Trim to max seconds if provided
    y = y[:int(max_seconds * sr)]
    
    if sr != target_sr:
        y = librosa.resample(y.T, orig_sr=sr, target_sr=target_sr).T
        sr = target_sr

    # Feature extraction with reduced frames
    hop_length = 512 # Smaller hop length for more frames
    n_mfcc = 20 # Number of MFCCs to extract, reduced for efficiency improvements
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Obtain energy, pitch, spectral centroid calculations from the audio signal
    energy = librosa.feature.rms(y=y, hop_length=hop_length)
    pitch = librosa.yin(y=y, fmin=50, fmax=300, frame_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)

    # Stack all features along rows (time frames) and then take mean across time to get fixed-size vector
    features = np.concatenate([
        mfcc.mean(axis=1), 
        mfcc_delta.mean(axis=1), 
        mfcc_delta2.mean(axis=1), 
        energy.mean(axis=1),
        [pitch.mean()],
        [spectral_centroid.mean()]
    ])
    
    return features