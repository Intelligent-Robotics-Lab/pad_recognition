# Yet to be updated for the new architecture and libraries, but will serve as a demo script to show how to load the 
# trained models and run inference on a sample from the test set. This can be expanded to run on multiple samples or 
# the entire test set if desired.

import joblib
import time
import numpy as np
from train import load_ravdess_data, load_isear
from models.fusion_model import fuse_pad
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def main():
    print("Multimodal Emotion Recognition Demo")

    # Load in the trained models

    try:
        text_model = joblib.load("saved_models/text_regressor_model.joblib")
        audio_model = joblib.load("saved_models/audio_regressor_model.joblib")
        video_model = joblib.load("saved_models/video_regressor_model.joblib")
    except Exception as e:
        print("Error loading models: ", e)
        return
    
    # Load in the data as used in the train and infer files
    df_text = load_isear(sample_size=2500)
    texts = df_text["sentence"].tolist()
    pad_targets_text = df_text["PAD"].tolist()
    _, test_texts, _, test_targets_text = train_test_split(
        texts, pad_targets_text, test_size=0.2, random_state=42)

    audio_paths, pad_targets_audio = load_ravdess_data("data/audio_data", [".wav"])
    _, test_audios, _, test_targets_audio = train_test_split(
        audio_paths, pad_targets_audio, test_size=0.2, random_state=42)
    
    video_paths, pad_targets_video = load_ravdess_data("data/video_data", [".mp4"])
    _, test_videos, _, test_targets_video = train_test_split(
        video_paths, pad_targets_video, test_size=0.2, random_state=42)

    # Demo the first sample only
    i = 0

    # Text
    example_text = test_texts[i]
    actual_text = test_targets_text[i]
    pred_text = text_model.predict([example_text])[0]

    # Audio
    example_audio = test_audios[i]
    actual_audio = test_targets_audio[i]
    pred_audio = audio_model.predict(example_audio)[0]

    # Video
    example_video = test_videos[i]
    actual_video = test_targets_video[i]
    pred_video = video_model.predict(example_video)[0]

    # Fuse predictions
    fused_pred = fuse_pad([pred_text, pred_audio, pred_video])
    fused_actual = np.mean([actual_text, actual_audio, actual_video], axis=0)

    # Print results
    print(f"Text PAD:   Predicted {pred_text}, Actual {actual_text}")
    print(f"Audio PAD:  Predicted {pred_audio}, Actual {actual_audio}")
    print(f'Video PAD:  Predicted {pred_video}, Actual {actual_video}')
    print(f"Fused PAD:  Predicted {fused_pred}, Actual {fused_actual}")
    print(f"Fused MAE:  {mean_absolute_error(fused_actual, fused_pred):.4f}")

    print("Demo complete. Additional examples can be evaluated similarly if desired.")


if __name__ == "__main__":
    main()