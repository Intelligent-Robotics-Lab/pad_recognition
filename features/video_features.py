import cv2
import numpy as np
import torch

from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Face detector
mtcnn = MTCNN(
    keep_all=False,
    post_process=False,
    min_face_size=40,
    device=device,
)

# Pretrained model used as feature extractor
fer = HSEmotionRecognizer(
    model_name="enet_b0_8_best_afew",
    device=device,
)


def extract_video_features(video_path, start_time, end_time, frame_skip=5):

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    features = []

    frame_idx = start_frame

    while True:
        
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if current_frame >= end_frame:
            break

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        frame_idx += 1

        # OpenCV uses BGR but models expect RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        boxes, probs = mtcnn.detect(frame, landmarks=False)

        if boxes is None:
            continue

        valid = probs > 0.90

        if not np.any(valid):
            continue

        # Crop the highest-confidence face
        box = boxes[valid][0].astype(int)

        x1, y1, x2, y2 = box

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Run on the pretrained HSEmotion backbone
        embedding = fer.extract_features(face)

        features.append(embedding.squeeze())

    cap.release()

    # If no faces were found
    if len(features) == 0:
        return np.zeros((1, 1280), dtype=np.float32)

    return np.stack(features).astype(np.float32)