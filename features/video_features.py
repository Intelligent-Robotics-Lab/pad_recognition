import cv2
import numpy as np
import torch
import face_alignment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the face-alignment model (detects facial landmarks)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)

# Optionally, choose which AUs you want
selected_aus = [
    'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17',
    'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU43'
]

def landmarks_to_features(landmarks):
    """Convert facial landmarks to a feature vector that includes AU-style proxies and head pose features."""
    features = []

    # If no landmarks detected, return zeros
    if landmarks is None or len(landmarks) == 0:
        return np.zeros(len(selected_aus) + 3, dtype=np.float32)

    lm = landmarks[0]  # shape (68,2)

    # Calculating inter-ocular distance for normalization so there aren't drastic difference between close and far faces
    scale = np.linalg.norm(lm[36] - lm[45]) + 1e-6

    # **Important to confirm and refine these values before large running again**

    # AU-style proxies
    # 1-2: Brow raises (inner + outer)
    features.append(np.linalg.norm(lm[21] - lm[19]) / scale)  # AU01 Inner Brow Raise
    features.append(np.linalg.norm(lm[22] - lm[24]) / scale)  # AU02 Outer Brow Raise

    # 3: Brow lowerer (AU04)
    features.append(np.linalg.norm(lm[21] - lm[39]) / scale)  # left brow to eye
    # 4: Upper lid raiser (AU05)
    features.append(np.linalg.norm(lm[37] - lm[41]) / scale)  # left eye vertical aperture
    # 5: Cheek raiser (AU06)
    features.append(np.linalg.norm(lm[41] - lm[37]) / scale)  # left eye aperture again as proxy
    # 6: Lid tightener (AU07)
    features.append(np.linalg.norm(lm[41] - lm[37]) / scale)  # approximate angle/closure
    # 7: Nose wrinkle (AU09)
    features.append(np.linalg.norm(lm[31] - lm[35]) / scale)
    # 8: Upper lip raiser (AU10)
    features.append(np.linalg.norm(lm[51] - lm[62]) / scale)
    # 9: Lip corner puller / smile (AU12)
    features.append(np.linalg.norm(lm[54] - lm[48]) / scale)
    # 10: Dimpler (AU14)
    features.append(np.linalg.norm(lm[54] - lm[48]) / scale)
    # 11: Lip corner depressor (AU15)
    features.append(np.linalg.norm(lm[48] - lm[60]) / scale)
    # 12: Chin raiser (AU17)
    features.append(np.linalg.norm(lm[8] - lm[57]) / scale)
    # 13: Lip stretcher (AU20)
    features.append(np.linalg.norm(lm[54] - lm[48]) / scale)
    # 14: Lip tightener (AU23)
    features.append(np.linalg.norm(lm[48] - lm[60]) / scale)
    # 15: Lips part (AU25)
    features.append(np.linalg.norm(lm[62] - lm[66]) / scale)
    # 16: Jaw drop (AU26)
    features.append(np.linalg.norm(lm[8] - lm[57]) / scale)
    # 17: Lip suck (AU28)
    features.append(np.linalg.norm(lm[62] - lm[66]) / scale)
    # 18: Eyes closed / blink (AU43)
    features.append(np.linalg.norm(lm[37] - lm[41]) / scale)  # left eye aperture as proxy

    # Pad any remaining to reach 19 if needed (should all be covered)
    while len(features) < len(selected_aus):
        features.append(0.0)

    # Head pose proxies
    nose = lm[30]
    chin = lm[8]
    left_eye = lm[36]
    right_eye = lm[45]

    pitch = (chin[1] - nose[1]) / scale
    yaw = (right_eye[0] - left_eye[0]) / scale
    roll = (right_eye[1] - left_eye[1]) / scale

    features.extend([pitch, yaw, roll])

    return np.array(features, dtype=np.float32)

    # expected_dim = len(selected_aus) + 3
    # if features.shape[0] != expected_dim:
    #     padded = np.zeros(expected_dim, dtype=np.float32)
    #     padded[:min(len(features), expected_dim)] = features[:expected_dim]
    #     features = padded

def extract_video_features(video_path, sample_fps=5, max_seconds=None, resize_dim=(160,160), include_deltas=True):
    """
    Extract frame-level facial features using 2D landmarks.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    frame_features = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps / sample_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(min(total_frames, (max_seconds * fps) if max_seconds else total_frames))

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            try:
                # Resize and convert to RGB
                frame_rgb = cv2.cvtColor(cv2.resize(frame, resize_dim), cv2.COLOR_BGR2RGB)
                landmarks = fa.get_landmarks(frame_rgb)
                features = landmarks_to_features(landmarks)

                # Mark unreliable frames
                if np.sum(features) < 1e-6:
                    features[:] = np.nan

                frame_features.append(features)
            except Exception as e:
                print(f"Frame {frame_idx} skipped: {e}")
                frame_features.append(np.full(len(selected_aus) + 3, np.nan, dtype=np.float32))

        frame_idx += 1

    cap.release()
    frame_features = np.stack(frame_features).astype(np.float32)

    # Interpolate missing frames
    for d in range(frame_features.shape[1]):
        x = frame_features[:, d]
        nans = np.isnan(x)
        if np.any(nans):
            indices = np.arange(len(x))
            x[nans] = np.interp(indices[nans], indices[~nans], x[~nans])
            frame_features[:, d] = x

    # Normalize per video
    mean = np.mean(frame_features, axis=0, keepdims=True)
    std = np.std(frame_features, axis=0, keepdims=True) + 1e-6
    frame_features = (frame_features - mean) / std


    # **Important to revisit if results seem off**: Consider adding a small constant to std to avoid division by zero, especially if some features are constant across frames.
    # Optional temporal deltas
    if include_deltas:
        deltas = np.diff(frame_features, axis=0, prepend=frame_features[0:1])
        frame_features = np.concatenate([frame_features, deltas], axis=1)

    return frame_features