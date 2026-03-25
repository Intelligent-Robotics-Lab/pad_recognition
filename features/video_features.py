import cv2
import numpy as np
import torch
import face_alignment

# Returns tensor of shape (T_video, 52)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the face-alignment model (detects facial landmarks)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)

# Optionally, choose which AUs you want
selected_aus = [
    'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17',
    'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU43'
]

import numpy as np

selected_dim = 26  # static features; final will be 52 with deltas

def landmarks_to_features(landmarks):
    """
    Convert 68-point landmarks to a 26-dim geometric feature vector.
    AU-inspired, scale-normalized, non-redundant.
    """
    # If no landmarks, return zeros
    if landmarks is None or len(landmarks) == 0:
        return np.zeros(selected_dim, dtype=np.float32)

    lm = landmarks[0]  # (68, 2)

    # Inter-ocular distance for scale normalization
    left_eye_outer = lm[36]
    right_eye_outer = lm[45]
    scale = np.linalg.norm(left_eye_outer - right_eye_outer) + 1e-6

    feats = []

    # EYE FEATURES
    # Left eye: points 37 (upper), 41 (lower)
    left_eye_aperture = np.linalg.norm(lm[37] - lm[41]) / scale
    # Right eye: points 43 (upper), 47 (lower)
    right_eye_aperture = np.linalg.norm(lm[43] - lm[47]) / scale
    mean_eye_aperture = 0.5 * (left_eye_aperture + right_eye_aperture)
    eye_asym = left_eye_aperture - right_eye_aperture
    eye_closure = 1.0 / (mean_eye_aperture + 1e-6)  # higher when eyes closed
    eye_tension = 1.0 / (np.abs(eye_asym) + mean_eye_aperture + 1e-6)

    feats.extend([left_eye_aperture,right_eye_aperture,mean_eye_aperture,eye_asym,eye_closure,eye_tension])

    # BROW FEATURES
    # Inner brow vs eye center
    left_brow_inner = lm[21]
    right_brow_inner = lm[22]
    left_eye_center = 0.5 * (lm[37] + lm[41])
    right_eye_center = 0.5 * (lm[43] + lm[47])

    left_brow_raise = (left_brow_inner[1] - left_eye_center[1]) / scale  # vertical
    right_brow_raise = (right_brow_inner[1] - right_eye_center[1]) / scale
    brow_avg = 0.5 * (left_brow_raise + right_brow_raise)
    brow_asym = left_brow_raise - right_brow_raise

    feats.extend([left_brow_raise,right_brow_raise, brow_avg, brow_asym])

    # MOUTH SHAPE FEATURES
    left_mouth = lm[48]
    right_mouth = lm[54]
    upper_lip = lm[51]
    lower_lip = lm[57]
    inner_upper = lm[62]
    inner_lower = lm[66]
    nose_tip = lm[33]
    chin = lm[8]

    mouth_width = np.linalg.norm(right_mouth - left_mouth) / scale
    mouth_height = np.linalg.norm(inner_upper - inner_lower) / scale
    mouth_ratio = mouth_width / (mouth_height + 1e-6)

    upper_lip_raise = (nose_tip[1] - upper_lip[1]) / scale
    jaw_drop = (chin[1] - lower_lip[1]) / scale

    lip_corner_asym = (left_mouth[1] - right_mouth[1]) / scale

    outer_lip_dist = np.linalg.norm(upper_lip - lower_lip) / scale
    inner_lip_dist = np.linalg.norm(inner_upper - inner_lower) / scale
    lip_tightness = outer_lip_dist - inner_lip_dist

    # Smile curvature: angle at upper lip between corners
    v1 = left_mouth - upper_lip
    v2 = right_mouth - upper_lip
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    smile_curvature = cos_angle  # closer to -1 → wide smile

    feats.extend([mouth_width,mouth_height,mouth_ratio,upper_lip_raise,jaw_drop,lip_corner_asym,lip_tightness,smile_curvature])

    # NOSE / MIDFACE
    left_nose = lm[31]
    right_nose = lm[35]
    nose_width = np.linalg.norm(left_nose - right_nose) / scale

    mouth_center = 0.5 * (left_mouth + right_mouth)
    nose_to_upper_lip = (nose_tip[1] - upper_lip[1]) / scale
    nose_to_mouth_center = np.linalg.norm(nose_tip - mouth_center) / scale

    feats.extend([ nose_width,nose_to_upper_lip,nose_to_mouth_center])

    # HEAD POSE PROXIES
    # Pitch: chin vs nose vertical
    pitch = (chin[1] - nose_tip[1]) / scale
    # Yaw: horizontal offset between eyes
    yaw = (right_eye_outer[0] - left_eye_outer[0]) / scale
    # Roll: vertical difference between eyes
    roll = (right_eye_outer[1] - left_eye_outer[1]) / scale

    feats.extend([pitch,yaw,roll])

    # FACE SIZE / STABILITY
    top_face = lm[27]
    bottom_face = chin
    left_face = lm[0]
    right_face = lm[16]

    face_height = np.linalg.norm(top_face - bottom_face) / scale
    face_width = np.linalg.norm(left_face - right_face) / scale

    feats.extend([face_height,face_width])

    feats = np.array(feats, dtype=np.float32)

    # Safety: pad/truncate to selected_dim
    if feats.shape[0] < selected_dim:
        padded = np.zeros(selected_dim, dtype=np.float32)
        padded[:feats.shape[0]] = feats
        feats = padded
    elif feats.shape[0] > selected_dim:
        feats = feats[:selected_dim]

    return feats

def extract_video_features(video_path, sample_fps=5, max_seconds=None,
                           resize_dim=(160, 160), include_deltas=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps / sample_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(min(total_frames, (max_seconds * fps) if max_seconds else total_frames))

    frame_features = []
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            try:
                frame_rgb = cv2.cvtColor(
                    cv2.resize(frame, resize_dim),
                    cv2.COLOR_BGR2RGB
                )
                landmarks = fa.get_landmarks(frame_rgb)
                feats = landmarks_to_features(landmarks)

                if np.sum(np.abs(feats)) < 1e-6:
                    feats[:] = np.nan

                frame_features.append(feats)
            except Exception:
                frame_features.append(np.full(selected_dim, np.nan, dtype=np.float32))

        frame_idx += 1

    cap.release()

    if len(frame_features) == 0:
        return np.zeros((1, selected_dim * (2 if include_deltas else 1)), dtype=np.float32)

    frame_features = np.stack(frame_features).astype(np.float32)  # (T, 26)

    # Interpolate NaNs
    for d in range(frame_features.shape[1]):
        x = frame_features[:, d]
        nans = np.isnan(x)
        if np.any(nans):
            idx = np.arange(len(x))
            valid = idx[~nans]
            if len(valid) == 0:
                x[:] = 0.0
            else:
                x[nans] = np.interp(idx[nans], valid, x[valid])
            frame_features[:, d] = x

    # Normalize per video
    mean = np.mean(frame_features, axis=0, keepdims=True)
    std = np.std(frame_features, axis=0, keepdims=True) + 1e-6
    std = np.maximum(std, 1e-3)
    frame_features = (frame_features - mean) / std

    # Add deltas
    if include_deltas:
        deltas = np.diff(frame_features, axis=0, prepend=frame_features[0:1])
        frame_features = np.concatenate([frame_features, deltas], axis=1)  # (T, 52)

    return frame_features.astype(np.float32)