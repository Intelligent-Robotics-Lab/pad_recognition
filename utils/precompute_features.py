import os
import torch
from tqdm import tqdm

from utils.helpers import MELDMultimodalDataset
from features.text_features import prepare_text_features
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features


CHUNK_SIZE = 1000  # Increased for efficiency
PRECOMPUTE_DIR = "data/MELD.Raw/precomputed_v2"


def get_existing_chunks(split):
    files = [
        f for f in os.listdir(PRECOMPUTE_DIR)
        if f.startswith(f"{split}_features_")
    ]

    # Sort numerically instead of lexicographically
    files.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
    return files


def precompute(split, max_samples=None):
    dataset = MELDMultimodalDataset(
        root_dir="/home/carter/pad_recognition/data/MELD.Raw",
        split=split
    )

    os.makedirs(PRECOMPUTE_DIR, exist_ok=True)

    chunk_data = []
    start_idx = 0
    chunk_idx = 0

    # Resume logic (disabled if max_samples is set)
    if max_samples is None:
        existing_chunks = get_existing_chunks(split)

        if existing_chunks:
            last_chunk_file = existing_chunks[-1]
            last_chunk_idx = int(last_chunk_file.split("_")[-1].replace(".pt", ""))

            chunk_idx = last_chunk_idx + 1
            start_idx = chunk_idx * CHUNK_SIZE

            print(f"Resuming {split} from sample {start_idx} (chunk {chunk_idx})")
    else:
        print(f"Extracting only {max_samples} samples - resume disabled.")

    total_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for idx in tqdm(range(start_idx, total_samples), desc=f"Extracting {split} features"):
        text, audio_path, video_path, pad_target = dataset[idx]

        # -------- Feature Extraction --------
        try:
            text_feat = prepare_text_features(text).cpu()
        except Exception as e:
            print(f"[Warning] Text failed @ {idx}: {e}")
            text_feat = torch.zeros(768)

        try:
            audio_feat = torch.as_tensor(
                extract_audio_features(audio_path),
                dtype=torch.float32
            )
        except Exception as e:
            print(f"[Warning] Audio failed @ {idx}: {e}")
            audio_feat = torch.zeros(63)

        try:
            video_feat = torch.as_tensor(
                extract_video_features(video_path),
                dtype=torch.float32
            )
        except Exception as e:
            print(f"[Warning] Video failed @ {idx}: {e}")
            video_feat = torch.zeros(52)

        sample = {
            "text": text_feat,
            "audio": audio_feat,
            "video": video_feat,
            "pad": torch.tensor(pad_target, dtype=torch.float32),
        }

        chunk_data.append(sample)

        # -------- Save full chunk --------
        if len(chunk_data) == CHUNK_SIZE:
            chunk_file = os.path.join(
                PRECOMPUTE_DIR,
                f"{split}_features_{chunk_idx}.pt"
            )

            torch.save(chunk_data, chunk_file)
            print(f"Saved chunk {chunk_idx} ({len(chunk_data)} samples)")

            chunk_data = []
            chunk_idx += 1

        # -------- Progress logging --------
        if idx % 500 == 0 and idx > 0:
            print(f"Processed {idx} / {total_samples} samples...")

    # -------- Save remaining samples --------
    if chunk_data:
        chunk_file = os.path.join(
            PRECOMPUTE_DIR,
            f"{split}_features_{chunk_idx}.pt"
        )

        torch.save(chunk_data, chunk_file)
        print(f"Saved final chunk {chunk_idx} ({len(chunk_data)} samples)")


if __name__ == "__main__":
    precompute("train")
    precompute("test")
    precompute("dev")