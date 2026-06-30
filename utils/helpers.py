import os
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import subprocess 

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

def get_emotions_indices(pad_targets):
    # Define PAD centers as tensor (NOT dict)
    emotion_to_pad = torch.tensor([
        [-0.51,  0.59,  0.25],   # anger
        [-0.375, 0.1,   0.15],   # disgust
        [-0.64,  0.6,  -0.43],   # fear
        [ 0.4,   0.2,   0.1],    # joy
        [ 0.0,   0.0,   0.0],    # neutral
        [-0.34, -0.1,  -0.52],   # sadness
        [ 0.6,   0.6,   0.4],    # surprise
    ], device=pad_targets.device)

    dists = torch.cdist(pad_targets, emotion_to_pad)  # (batch, 7)
    return torch.argmin(dists, dim=1)

# Construct paths for audio/video files in the train/dev/test subfolders
def build_media_paths(root_dir, split, dialogue_id, utterance_id):
    clip_name = f"dia{dialogue_id}_utt{utterance_id}"
    folder = os.path.join(root_dir, split)
    video_path = os.path.join(folder, clip_name + ".mp4")
    audio_path = os.path.join(folder, clip_name + ".wav")
    return audio_path, video_path

# Dataset class for storing dataset statistics (mean/std for text/audio/video) used for normalization in the PrecomputedDataset
class DatasetStats:
    def __init__(self, stats_path):
        stats = torch.load(stats_path)

        self.text_mean = stats["text_mean"]
        self.text_std = stats["text_std"]

        self.audio_mean = stats["audio_mean"]
        self.audio_std = stats["audio_std"]

        self.video_mean = stats["video_mean"]
        self.video_std = stats["video_std"]

# Dataset class for loading precomputed tensors (text/audio/video) and PAD targets, with optional normalization
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

        print(f"DEBUG: idx={idx}, emotion='{emotion}', pad_target={pad_target}")

        return text, audio_path, video_path, pad_target

# Precomputed dataset class for loading tensors from disk, with optional normalization using provided statistics
class PrecomputedDataset(Dataset):
    def __init__(self, file_path, stats_path=None):
        self.data = torch.load(file_path)
        self.stats = DatasetStats(stats_path) if stats_path else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        text = sample["text"]
        audio = sample["audio"]
        video = sample["video"]
        pad =sample["pad"]
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(0)
        assert audio.dim() == 2

        if video.dim() == 1:
            video = video.unsqueeze(0)
        elif video.dim() == 3:
            video = video.squeeze(0)
        assert video.dim() == 2

        if self.stats is not None:
            text = (text - self.stats.text_mean) / (self.stats.text_std + 1e-8)
            audio = (audio - self.stats.audio_mean) / (self.stats.audio_std + 1e-8)
            video = (video - self.stats.video_mean) / (self.stats.video_std + 1e-8)
        
        return text, audio, video, pad

# Collate function to handle variable-length sequences and filter out invalid samples
def multimodal_collate(batch):
    text, audio, video, pad = zip(*batch)

    expected_audio_dim = 1024
    expected_video_dim = 7

    filtered_text, filtered_audio, filtered_video, filtered_pad = [], [], [], []

    for t, a, v, y in zip(text, audio, video, pad):
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if v.ndim == 1:
            v = v.unsqueeze(0)

        # if a.shape[1] != expected_audio_dim:
        #     continue
        if v.shape[1] != expected_video_dim:
            continue

        if torch.all(a == 0) or torch.all(v == 0):
            continue

        filtered_text.append(t)
        filtered_audio.append(a)
        filtered_video.append(v)
        filtered_pad.append(y)

    if len(filtered_text) == 0:
        return None

    text = pad_sequence(filtered_text, batch_first=True)     # [B, T_text, 1024]
    audio = pad_sequence(filtered_audio, batch_first=True)   # [B, T_audio, 1024]
    # audio_lengths = torch.tensor([a.shape[0] for a in filtered_audio])
    video = pad_sequence(filtered_video, batch_first=True)   # [B, T_video, 7]
    pad = torch.stack(filtered_pad)                          # [B, 3]

    return text, audio, video, pad

# Extract audio from .mp4 files using moviepy (fallback method, less efficient than ffmpeg) 
def extract_audio_from_mp4(mp4_path, save_path, target_sr=16000):
    if os.path.exists(save_path):
        print(f"Skipped (already exists): {save_path}")
        return save_path

    clip = VideoFileClip(mp4_path)

    clip.audio.write_audiofile(save_path, fps=target_sr, verbose=False, logger=None)
    clip.close()
    print(f"Saved: {save_path}")
    return save_path

# Audio extraction using ffmpeg for better performance and reliability compared to moviepy
def extract_audio_ffmpeg(mp4_path, wav_path, target_sr=16000):
    if os.path.exists(wav_path):
        print(f"Skipped (already exists): {wav_path}")
        return wav_path

    ffmpeg_path = r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

    cmd = [
        ffmpeg_path,
        "-i", mp4_path,         # input file
        "-vn",                  # ignore video
        "-ac", "1",             # convert to mono
        "-ar", str(target_sr),  # sampling rate
        "-y",                   # overwrite if exists
        wav_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Saved: {wav_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {mp4_path}: {e}")

    return wav_path

# Walk through MELD train/dev/test folders and extract audio for all .mp4 files
def extract_all_meld_audio(root_dir):
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