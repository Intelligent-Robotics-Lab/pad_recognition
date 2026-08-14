"""
Precomputes video features for every utterance in data/iemocap.csv and writes a
video_feature_path column back into that CSV. Run once (and again any time
frame_skip in features/video_features.py changes) before using train/infer
scripts on IEMOCAP — IEMOCAPDataset loads features from disk rather than
re-extracting them on the fly.

Run from the repo root: PYTHONPATH=. python utils/precompute_video_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from features.video_features import extract_video_features

CSV_PATH = "data/iemocap.csv"
ROOT = Path("data/IEMOCAP")
OUTPUT_DIR = ROOT / "precomputed_video"


def main():
    df = pd.read_csv(CSV_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_feature_paths = []

    for i, row in df.iterrows():

        rel_path = f"precomputed_video/{row['utterance_id']}.npy"
        out_path = ROOT / rel_path

        if out_path.exists():
            video_feature_paths.append(rel_path)
            continue

        video_path = ROOT / row["video_path"]

        feats = extract_video_features(video_path, row["start_time"], row["end_time"])

        assert feats.shape[-1] == 1280, (
            f"Expected 1280-dim HSEmotion features, got shape {feats.shape} "
            f"for {row['utterance_id']}"
        )

        np.save(out_path, feats)

        video_feature_paths.append(rel_path)

        if i % 100 == 0:
            print(f"{i}/{len(df)} — {row['utterance_id']}")

    df["video_feature_path"] = video_feature_paths

    df.to_csv(CSV_PATH, index=False)

    print(f"Done. Wrote features to {OUTPUT_DIR}, updated {CSV_PATH}.")


if __name__ == "__main__":
    main()
