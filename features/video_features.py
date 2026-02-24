import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
# from fer import FER

# Load MobileNetV2 without the top layer, use as feature extractor
feature_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
# Output vector will be 1280-d

# Face detector (optional)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_video_features(video_path, sample_fps=3, max_seconds=None, resize_dim=(160,160)):
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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, resize_dim)
                    face_resized = preprocess_input(face_resized.astype(np.float32))
                    face_resized = np.expand_dims(face_resized, axis=0)

                    features = feature_model.predict(face_resized, verbose=0)[0]
                    frame_features.append(features)

            except Exception as e:
                print(f"Frame {frame_idx} skipped: {e}")
                pass

        frame_idx += 1

    cap.release()

    if len(frame_features) == 0:
        return np.zeros(feature_model.output_shape[1],)

    # Average across frames
    video_feature_vector = np.mean(frame_features, axis=0)
    return video_feature_vector

# Original with FER emotion detection, which is more interpretable but wasn't working due to package error

# detector = FER()

# def extract_video_features(video_path, sample_fps=5, max_seconds=None, resize_dim=(160,160)):
#     # Read in and break into frames
#     cap = cv2.VideoCapture(video_path)
#     frame_features = []

#     # Only sampling 5fps for sake of efficiency, increases latency but should still capture emotional content well for short clips
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         fps = 30 # Default if unknown
#     frame_interval = max(1, int(fps / sample_fps))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     max_frames = int(min(total_frames, (max_seconds * fps) if max_seconds else total_frames))
    
#     frame_idx = 0
#     while frame_idx < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_idx % frame_interval == 0:
#             try:
#                 frame_resized = cv2.resize(frame, resize_dim)
#                 results = detector.detect_emotions(frame_resized)
#                 if results:
#                     emotions = results[0]["emotions"]
#                     emotion_vector = np.array(list(emotions.values()))
#                     frame_features.append(emotion_vector)
#             except:
#                 pass 
            
#         frame_idx += 1

#     # Closes the video file
#     cap.release()

#     if len(frame_features) == 0:
#         return np.zeros((7,))  # Assuming 7 emotion categories
        
#     # Take the mean emotion probabilities across all frames giving one feature vector for the video clip
#     processed_frames = np.mean(frame_features, axis=0)

#     # This gives a 7-d emotion vector that reflects the overall clip
#     return processed_frames