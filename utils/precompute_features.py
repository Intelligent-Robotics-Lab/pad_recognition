# Needs to be updated and finished, haven't implemented currently

from torch.utils.data import Dataset
import torch
import os
from utils.helpers import MELDMultimodalDataset
from features.text_features import prepare_text_features
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features

class MELDPrecompDataset(Dataset):
    def __init__(self, precomp_dir, split="train"):
        self.text_dir = os.path.join(precomp_dir, "text")
        self.audio_dir = os.path.join(precomp_dir, "audio")
        self.video_dir = os.path.join(precomp_dir, "video")
        self.pad_dir = os.path.join(precomp_dir, "pad_targets")
        self.length = len(os.listdir(self.text_dir))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        text_feat = torch.load(os.path.join(self.text_dir, f"{idx}.pt"))
        audio_feat = torch.load(os.path.join(self.audio_dir, f"{idx}.pt"))
        video_feat = torch.load(os.path.join(self.video_dir, f"{idx}.pt"))
        pad_target = torch.load(os.path.join(self.pad_dir, f"{idx}.pt"))
        return text_feat, audio_feat, video_feat, pad_target