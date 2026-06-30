import os
import torch

PRECOMPUTE_DIR = "data/MELD.Raw/precomputed_v4"

def load_all_chunks(split):
    files = [
        f for f in os.listdir(PRECOMPUTE_DIR)
        if f.startswith(f"{split}_features_")
    ]
    files.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))

    all_samples = []
    for f in files:
        path = os.path.join(PRECOMPUTE_DIR, f)
        all_samples.extend(torch.load(path))

    return all_samples

def collect_modalities(samples):
    text_all, audio_all, video_all = [], [], []

    for s in samples:
        text = s["text"]
        audio = s["audio"]
        video = s["video"]

        if text.dim() == 2:
            text = text.mean(dim=0)
        text_all.append(text)

        if audio.dim() == 2:
            audio_all.append(audio.reshape(-1, audio.shape[-1]))

        if video.dim() == 2:
            video_all.append(video.reshape(-1, video.shape[-1]))

    text_all = torch.stack(text_all)
    audio_all = torch.cat(audio_all, dim=0)
    video_all = torch.cat(video_all, dim=0)

    return text_all, audio_all, video_all

def compute_stats(x):
    mean = x.mean(dim=0)
    std = x.std(dim=0) + 1e-8
    return mean, std

def main():
    print("Loading train data...")
    samples = load_all_chunks("train")

    print("Collecting modalities...")
    text, audio, video = collect_modalities(samples)

    print("Computing stats...")

    text_mean, text_std = compute_stats(text)
    audio_mean, audio_std = compute_stats(audio)
    video_mean, video_std = compute_stats(video)

    stats = {
        "text_mean": text_mean,
        "text_std": text_std,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
        "video_mean": video_mean,
        "video_std": video_std,
    }

    torch.save(stats, "data/MELD.Raw/precomputed_v4/dataset_stats.pt")

    print("Saved dataset_stats.pt")

if __name__ == "__main__":
    main()