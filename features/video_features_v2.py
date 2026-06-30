import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")  # Disable GPU for this script (had to be done before importing FER)

import cv2
import numpy as np
from fer import FER

emotion_keys = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
n_emotions = len(emotion_keys)
emotion_detector = FER(mtcnn=True)

def extract_emotion_probs(
    video_path: str, 
    sample_fps: int=3, 
    include_deltas: bool=False,
    min_valid_ratio: float=0.1,
    ema_alpha: float=0.85,
) -> np.ndarray:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(native_fps / sample_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_out = n_emotions * (2 if include_deltas else 1)

    features = []
    valid_count = 0
    frame_idx = 0

    while True:
        if frame_idx % frame_interval != 0:
            if not cap.grab():
                break
            frame_idx += 1
            continue
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            detections = emotion_detector.detect_emotions(frame_rgb)
        except Exception:
            detections = []

        if detections:
            emotions = detections[0]["emotions"]
            vec = np.array([emotions.get(key, 0.0) for key in emotion_keys], dtype=np.float32)
            valid_count += 1
        else:
            vec = np.full(n_emotions, np.nan, dtype=np.float32)

        features.append(vec)
        frame_idx += 1
    
    cap.release()

    n_frames = len(features)
    if n_frames == 0:
        return np.zeros((1, n_out), dtype=np.float32)

    features = np.stack(features) # (T, 7)

    valid_ratio = valid_count / n_frames
    if valid_ratio < min_valid_ratio:
        return np.zeros((1, n_out), dtype=np.float32)

    # Vectorized NaN interpolation
    idx = np.arange(features.shape[0])
    for d in range(n_emotions):
        x = features[:, d]
        nans = np.isnan(x)
        if nans.any():
            valid_idx = idx[~nans]
            x[nans] = np.interp(idx[nans], valid_idx, x[valid_idx]) if len(valid_idx) else 0.0

    features = np.clip(features, 0.0, 1.0)

    # Vectorized EMA via scipy
    from scipy.signal import lfilter
    b = [ema_alpha]
    a = [1.0, -(1.0 - ema_alpha)]
    features = lfilter(b, a, features, axis=0).astype(np.float32)

    if include_deltas:
        deltas = np.diff(features, axis=0, prepend=features[0:1])
        features = np.concatenate([features, deltas], axis=1) # (T, 14)

    return features.astype(np.float32) # (T_video, 7) or 14 with deltas
