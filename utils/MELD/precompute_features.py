import os
import torch
from tqdm import tqdm
from utils.helpers import MELDMultimodalDataset
from features.text_features import prepare_text_features
from features.audio_features_v2 import extract_wav2vec_features
from features.video_features_v2 import extract_emotion_probs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHUNK_SIZE = 1000 
PRECOMPUTE_DIR = "data/MELD.Raw/precomputed_v4"
ROOT_DIR = "/home/carter/pad_recognition/data/MELD.Raw"

text_dim = 1024
audio_dim = 1024
video_dim = 7

def get_existing_chunks(split):
    files = [
        f for f in os.listdir(PRECOMPUTE_DIR)
        if f.startswith(f"{split}_features_")
    ]

    files.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", ""))) # Sort by chunk index
    return files

# Precompute features for a given split and save in chunks, with resume capability
def precompute(split, max_samples=None):
    dataset = MELDMultimodalDataset(
        root_dir=ROOT_DIR,
        split=split
    )

    os.makedirs(PRECOMPUTE_DIR, exist_ok=True)

    chunk_data = []
    start_idx = 0
    chunk_idx = 0

    total_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

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

    for idx in tqdm(range(start_idx, total_samples), desc=f"Extracting {split} features"):
        text, audio_path, video_path, pad_target = dataset[idx]

        try:
            text_feat = prepare_text_features(text).to(device)
        except Exception as e:
            print(f"[Warning] Text failed @ {idx}: {e}")
            text_feat = torch.zeros((1, text_dim))

        try:
            audio_feat = torch.as_tensor(
                extract_wav2vec_features(audio_path),
                dtype=torch.float32
            )
        except Exception as e:
            print(f"[Warning] Audio failed @ {idx}: {e}")
            audio_feat = torch.zeros((1, audio_dim))

        try:
            video_feat = torch.as_tensor(
                extract_emotion_probs(video_path),
                dtype=torch.float32
            )
        except Exception as e:
            print(f"[Warning] Video failed @ {idx}: {e}")
            video_feat = torch.zeros((1, video_dim))

        sample = {
            "text": text_feat.cpu(),
            "audio": audio_feat.cpu(),
            "video": video_feat.cpu(),
            "pad": torch.tensor(pad_target, dtype=torch.float32),
        }

        chunk_data.append(sample)

        # Save full chunk
        if len(chunk_data) == CHUNK_SIZE:
            chunk_file = os.path.join(
                PRECOMPUTE_DIR,
                f"{split}_features_{chunk_idx}.pt"
            )
            
            torch.save(chunk_data, chunk_file)
            print(f"Saved chunk {chunk_idx} ({len(chunk_data)} samples)")

            chunk_data = []
            chunk_idx += 1

        # Progress logging
        if idx % 500 == 0 and idx > 0:
            print(f"Processed {idx} / {total_samples} samples...")

    # Save remaining samples
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