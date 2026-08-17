import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import soundfile as sf

# Map MELD emotion labels → PAD values (taken directly from literature, see README for sources)
emotion_to_pad = {
    "anger":      [-0.51, 0.59,  0.25],
    "disgust":    [-0.375,  0.1,  0.15],    # average of dislike, hate, reproach, resentment
    "fear":       [-0.64,  0.6,  -0.43],
    "joy":        [ 0.4,  0.2,  0.1],
    "neutral":    [ 0.0,  0.0,  0.0],
    "sadness":    [-0.34,  -0.1,  -0.52],   # average of disappointment, distress, pity, remorse, shame
    "surprise":   [ 0.6,  0.6, 0.4]         # taken from the words section
}

# Construct paths for audio/video files in the train/dev/test subfolders
def build_media_paths(root_dir, split, dialogue_id, utterance_id):
    clip_name = f"dia{dialogue_id}_utt{utterance_id}"
    folder = os.path.join(root_dir, split)
    video_path = os.path.join(folder, clip_name + ".mp4")
    audio_path = os.path.join(folder, clip_name + ".wav")
    return audio_path, video_path

# Dataset class for on-the-fly MELD loading, mirroring IEMOCAPDataset (utils/iemocap_dataset.py):
# raw text/waveform/video-path per sample, features extracted per-batch during train/inference
# rather than precomputed and cached.
class MELDDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir: base folder containing `{split}_sent_emo.csv` and the `train`/`dev`/`test` media folders
        split: "train", "dev", or "test"
        """
        self.root_dir = root_dir
        file_map = {
            "train": "train_sent_emo.csv",
            "dev":   "dev_sent_emo.csv",
            "test":  "test_sent_emo.csv",
        }
        csv_path = os.path.join(root_dir, file_map[split])
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Emotion"].notna()].reset_index(drop=True)
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["Utterance"]
        emotion = row["Emotion"].lower()

        dialogue_id = row["Dialogue_ID"]
        utt_id = row["Utterance_ID"]

        audio_path, video_path = build_media_paths(self.root_dir, self.split, dialogue_id, utt_id)
        pad_target = torch.tensor(emotion_to_pad.get(emotion, [0.0, 0.1, 0.5]), dtype=torch.float32)

        waveform, sr = sf.read(audio_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # MELD clips are already trimmed to one utterance each (unlike IEMOCAP, where
        # start/end mark a sub-span of a longer session recording), so the full clip
        # duration is the end_time extract_video_features needs.
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        end_time = frame_count / fps if fps else 0.0

        return {
            "text": text,
            "audio": waveform,
            "sample_rate": sr,

            "video_path": video_path,

            "start_time": 0.0,
            "end_time": end_time,

            "pad": pad_target
        }
