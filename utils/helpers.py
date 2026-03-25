import os
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import subprocess 

# Map MELD emotion labels → PAD values
emotion_to_pad = {
    "anger":      [-0.51, 0.59,  0.25],
    "disgust":    [-0.375,  0.1,  0.15], # average of dislike, hate, reproach, resentment
    "fear":       [-0.64,  0.6,  -0.43],
    "joy":        [ 0.4,  0.2,  0.1],
    "neutral":    [ 0.0,  0.0,  0.0],
    "sadness":    [-0.34,  -0.1,  -0.52], # average of disappointment, distress, pity, remorse, shame
    "surprise":   [ 0.6,  0.6, 0.4] # taken from the words section
}

def build_media_paths(root_dir, split, dialogue_id, utterance_id):
    """
    Construct paths for audio/video files in the train/dev/test subfolders.
    """
    clip_name = f"dia{dialogue_id}_utt{utterance_id}"
    folder = os.path.join(root_dir, split)  # e.g., data/train, data/dev, data/test
    video_path = os.path.join(folder, clip_name + ".mp4")
    audio_path = os.path.join(folder, clip_name + ".wav")
    return audio_path, video_path

class MELDMultimodalDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        root_dir: base folder containing `train_sent_emo.csv`, `wav/`, and `mp4/`
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
        # filter out any missing labels
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

        # Pass split to build correct paths for audio/video files
        audio_path, video_path = build_media_paths(self.root_dir, self.split, dialogue_id, utt_id)
        pad_target = torch.tensor(emotion_to_pad.get(emotion, [0.0, 0.1, 0.5]), dtype=torch.float32)  # fallback PAD

        print(f"DEBUG: idx={idx}, emotion='{emotion}', pad_target={pad_target}")

        return text, audio_path, video_path, pad_target

class PrecomputedDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            sample["text"],
            sample["audio"],
            sample["video"],
            sample["pad"]
        )

import torch
from torch.nn.utils.rnn import pad_sequence

def multimodal_collate(batch):
    # batch is a list of (text, audio, video, pad)
    text, audio, video, pad = zip(*batch)

    expected_audio_dim = 63
    expected_video_dim = 52

    filtered_text, filtered_audio, filtered_video, filtered_pad = [], [], [], []

    for t, a, v, y in zip(text, audio, video, pad):
        # Ensure audio/video are 2D
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if v.ndim == 1:
            v = v.unsqueeze(0)

        # Dimension checks
        if a.shape[1] != expected_audio_dim:
            continue
        if v.shape[1] != expected_video_dim:
            continue

        # Skip only audio/video zero fallbacks
        if torch.all(a == 0) or torch.all(v == 0):
            continue

        filtered_text.append(t)
        filtered_audio.append(a)
        filtered_video.append(v)
        filtered_pad.append(y)

    # If nothing valid remains, skip batch
    if len(filtered_text) == 0:
        return None

    # Pad variable-length sequences
    text = pad_sequence(filtered_text, batch_first=True)     # [B, T_text, 768]
    audio = pad_sequence(filtered_audio, batch_first=True)   # [B, T_audio, 63]
    video = pad_sequence(filtered_video, batch_first=True)   # [B, T_video, 52]
    pad = torch.stack(filtered_pad)                          # [B, 3]

    return text, audio, video, pad
    
def extract_audio_from_mp4(mp4_path, save_path, target_sr=16000):
    """
    Extract audio from a .mp4 video and save as .wav.

    Parameters:
        mp4_path (str): Path to input .mp4 video
        save_path (str): Path to save .wav file
        target_sr (int): Sampling rate
    """
    # Skip if .wav already exists
    if os.path.exists(save_path):
        print(f"Skipped (already exists): {save_path}")
        return save_path

    clip = VideoFileClip(mp4_path)

    # Write audio to .wav with target sampling rate
    clip.audio.write_audiofile(save_path, fps=target_sr, verbose=False, logger=None)
    clip.close()
    print(f"Saved: {save_path}")
    return save_path

def extract_all_meld_audio(root_dir):
    """
    Walk through MELD train/dev/test folders and extract audio for all .mp4 files.
    
    Parameters:
        root_dir (str): path to MELD data folder containing train/dev/test
    """
    for split in ["train", "dev", "test"]:
        folder = os.path.join(root_dir, split)

        if not os.path.exists(folder):
            print(f"Folder does not exist, skipping: {folder}")
            continue

        print(f"\nProcessing {split} folder...")
        mp4_files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
        print(f"Found {len(mp4_files)} MP4 files in {folder}")  # debug

        for f in mp4_files:
            mp4_path = os.path.abspath(os.path.join(folder, f))
            wav_path = os.path.abspath(os.path.join(folder, f.replace(".mp4", ".wav")))

            if os.path.exists(wav_path):
                print("Already exists, skipping:", wav_path)
                continue

            try:
                extract_audio_ffmpeg(mp4_path, wav_path)
            except Exception as e:
                print(f"Error processing {mp4_path}: {e}")

def extract_audio_ffmpeg(mp4_path, wav_path, target_sr=16000):
    if os.path.exists(wav_path):
        print(f"Skipped (already exists): {wav_path}")
        return wav_path

    # Define the path
    ffmpeg_path = r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

    # ffmpeg command
    cmd = [
        ffmpeg_path,
        "-i", mp4_path,       # input file
        "-vn",                # ignore video
        "-ac", "1",           # convert to mono
        "-ar", str(target_sr),# sampling rate
        "-y",                 # overwrite if exists
        wav_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Saved: {wav_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {mp4_path}: {e}")

    return wav_path