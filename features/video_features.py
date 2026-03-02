# Would like to incorporate an LSTM model to gather trends across time by stacking emotion vecotrs

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
    'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU43'  # example main ones
]

def landmarks_to_features(landmarks):
    """Convert 68 facial landmarks to a simple AU-style vector."""
    features = []

    if landmarks is None or len(landmarks) == 0:
        return np.zeros(len(selected_aus) + 3, dtype=np.float32)

    lm = landmarks[0]  # shape (68,2)

    # AU-style proxies
    features.append(np.linalg.norm(lm[21] - lm[19]))  # brow raise L
    features.append(np.linalg.norm(lm[22] - lm[24]))  # brow raise R
    features.append(np.linalg.norm(lm[54] - lm[48]))  # smile width
    features.append(np.linalg.norm(lm[57] - lm[8]))   # jaw drop

    while len(features) < len(selected_aus):
        features.append(0.0)

    # Head pose proxies
    nose = lm[30]
    chin = lm[8]
    left_eye = lm[36]
    right_eye = lm[45]

    pitch = chin[1] - nose[1]
    yaw = right_eye[0] - left_eye[0]
    roll = right_eye[1] - left_eye[1]

    # Normalize head pose features as pitch/yaw/roll can be large pixel distances
    scale = np.linalg.norm(lm[36] - lm[45]) + 1e-6
    pitch /= scale
    yaw /= scale
    roll /= scale

    features.extend([pitch, yaw, roll])

    features = np.array(features, dtype=np.float32)

    expected_dim = len(selected_aus) + 3
    if features.shape[0] != expected_dim:
        padded = np.zeros(expected_dim, dtype=np.float32)
        padded[:min(len(features), expected_dim)] = features[:expected_dim]
        features = padded

    return features


def extract_video_features(video_path, sample_fps=5, max_seconds=None, resize_dim=(160,160)):
    cap = cv2.VideoCapture(video_path)
    frame_features = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

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
                landmarks = fa.get_landmarks(frame)
                features = landmarks_to_features(landmarks)

                if features.ndim != 1:
                    features = features.flatten()

                frame_features.append(features.astype(np.float32))

            except Exception as e:
                print(f"Frame {frame_idx} skipped: {e}")

        frame_idx += 1

    cap.release()

    base_dim = len(selected_aus) + 3

    if len(frame_features) == 0:
        return np.zeros(base_dim * 2, dtype=np.float32)

    frame_features = np.stack(frame_features)

    mean = np.mean(frame_features, axis=0)
    std = np.std(frame_features, axis=0)

    video_feature_vector = np.concatenate([mean, std]).astype(np.float32)

    print(f"Extracted video features shape: {video_feature_vector.shape}")

    return video_feature_vector