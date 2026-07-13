import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import soundfile as sf


class IEMOCAPDataset(Dataset):
    def __init__(self, csv_path, root="data/IEMOCAP"):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = row["text"]

        audio_path = self.root / row["audio_path"]

        waveform, sr = sf.read(audio_path)

        waveform = torch.tensor(
            waveform,
            dtype=torch.float32
        )

        pad = torch.tensor(
            [
                row["valence"],
                row["arousal"],
                row["dominance"]
            ],
            dtype=torch.float32
        )

        return {
            "text": text,
            "audio": waveform,
            "sample_rate": sr,
            "pad": pad
        }