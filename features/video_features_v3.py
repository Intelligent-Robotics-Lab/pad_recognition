import cv2
import numpy as np
import torch
from feat import Detector

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model="svm",
    emotion_model="resmasknet",
    facepose_model="img2pose",
    device=device,
)

au_cols = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10",
           "AU12","AU14","AU15","AU17","AU20","AU23","AU25","AU26","AU28","AU43"]
emotion_cols = ["anger","disgust","fear","happiness","sadness","surprise","neutral"]
pose_cols    = ["Pitch", "Roll", "Yaw"]
all_cols     = au_cols + emotion_cols + pose_cols   # 28 dims


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Per-channel linear interpolation over NaN rows."""
    for d in range(arr.shape[1]):
        x = arr[:, d]
        nans = np.isnan(x)
        if nans.any():
            idx = np.arange(len(x))
            valid = idx[~nans]
            x[nans] = np.interp(idx[nans], valid, x[valid]) if len(valid) else 0.0
            arr[:, d] = x
    return arr


def extract_pyfeat_features(
    video_path: str,
    sample_fps: int = 3,
    include_deltas: bool = True,
    min_valid_ratio: float = 0.10,
) -> tuple[torch.Tensor, dict]:

    n_out = len(all_cols) * (2 if include_deltas else 1)
    _bad  = (torch.zeros((1, n_out), device=device),
             {"valid_ratio": 0.0, "columns": all_cols})

    # Derive skip_frames from video's native FPS
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    skip_frames = max(0, int(native_fps / sample_fps) - 1)

    try:
        fex = detector.detect(
            video_path,
            data_type="video",
            skip_frames=skip_frames,
            progress_bar=False,
        )
    except Exception as e:
        print(f"[py-feat] failed on {video_path}: {e}")
        return _bad

    if fex is None or len(fex) == 0:
        return _bad

    # Pull only columns that exist in this fex result
    cols = [c for c in all_cols if c in fex.columns]
    data = np.full((len(fex), len(all_cols)), np.nan, dtype=np.float32)
    for i, c in enumerate(all_cols):
        if c in fex.columns:
            data[:, i] = fex[c].values.astype(np.float32)

    # Validity: rows where every AU is NaN = no face detected
    valid_mask  = ~np.all(np.isnan(data), axis=1)
    valid_ratio = valid_mask.mean()
    if valid_ratio < min_valid_ratio:
        return _bad[0], {"valid_ratio": float(valid_ratio), "columns": all_cols}

    data = _interpolate_nans(data)
    data = np.clip(data, 0.0, None)

    if include_deltas:
        deltas = np.diff(data, axis=0, prepend=data[0:1])
        data   = np.concatenate([data, deltas], axis=1)  # (T, 56)

    delta_cols = [f"Δ{c}" for c in all_cols] if include_deltas else []
    meta = {
        "valid_ratio": float(valid_ratio),
        "columns":     all_cols + delta_cols,
    }

    return torch.tensor(data, dtype=torch.float32, device=device), meta