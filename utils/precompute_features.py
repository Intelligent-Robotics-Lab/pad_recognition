import os
import torch
from tqdm import tqdm
from utils.helpers import MELDMultimodalDataset
from features.text_features import prepare_text_features
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHUNK_SIZE = 500  # number of samples per file
PRECOMPUTE_DIR = "data/MELD.Raw/precomputed"

def precompute(split):
    dataset = MELDMultimodalDataset(
        root_dir="/home/carter/pad_recognition/data/MELD.Raw",
        split=split
    )

    os.makedirs(PRECOMPUTE_DIR, exist_ok=True)

    chunk_data = []
    start_idx = 0

    # Detect last saved chunk
    existing_chunks = [
        f for f in os.listdir(PRECOMPUTE_DIR) if f.startswith(f"{split}_features_")
    ]
    if existing_chunks:
        existing_chunks.sort()
        last_chunk_file = existing_chunks[-1]
        last_chunk_idx = int(last_chunk_file.split("_")[-1].replace(".pt", ""))
        start_idx = (last_chunk_idx + 1) * CHUNK_SIZE
        print(f"Resuming {split} from sample {start_idx}")

    for idx in tqdm(range(start_idx, len(dataset)), desc=f"Extracting {split} features"):
        text, audio_path, video_path, pad_target = dataset[idx]

        # Safe feature extraction
        try:
            text_feat = prepare_text_features(text).cpu()
        except Exception as e:
            print(f"[Warning] Text feature failed at idx {idx}: {e}")
            text_feat = torch.zeros(768)

        try:
            audio_feat = torch.as_tensor(extract_audio_features(audio_path), dtype=torch.float32)
        except Exception as e:
            print(f"[Warning] Audio feature failed at idx {idx}: {e}")
            audio_feat = torch.zeros(63)

        try:
            video_feat = torch.as_tensor(extract_video_features(video_path), dtype=torch.float32)
        except Exception as e:
            print(f"[Warning] Video feature failed at idx {idx}: {e}")
            video_feat = torch.zeros(21)

        sample = {
            "text": text_feat,
            "audio": audio_feat,
            "video": video_feat,
            "pad": torch.tensor(pad_target, dtype=torch.float32)
        }

        chunk_data.append(sample)

        # Save chunk if full
        if len(chunk_data) == CHUNK_SIZE:
            chunk_idx = idx // CHUNK_SIZE
            chunk_file = os.path.join(PRECOMPUTE_DIR, f"{split}_features_{chunk_idx}.pt")
            torch.save(chunk_data, chunk_file)
            print(f"Saved chunk {chunk_idx} ({len(chunk_data)} samples)")
            chunk_data = []

        # Optional: print progress every 500 samples
        if idx % 500 == 0 and idx > 0:
            print(f"Processed {idx} / {len(dataset)} samples...")

    # Save any remaining samples (last partial chunk)
    if chunk_data:
        chunk_idx = len(dataset) // CHUNK_SIZE
        chunk_file = os.path.join(PRECOMPUTE_DIR, f"{split}_features_{chunk_idx}.pt")
        torch.save(chunk_data, chunk_file)
        print(f"Saved final chunk {chunk_idx} ({len(chunk_data)} samples)")

if __name__ == "__main__":
    precompute("train")
    precompute("test")