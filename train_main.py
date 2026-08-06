"""
Single entry point for training. Edit DATASET and MODE below, then run:
    python train_main.py

This only picks which script to run — modality/fusion choices are still set
by editing the variable near the top of the target script (MODALITY,
FUSION_TYPE), exactly as before.
"""

import os
import subprocess
import sys

# Options: "IEMOCAP" or "MELD"
DATASET = "IEMOCAP"

# Options: "ind", "ta", "multimodal"
MODE = "ind"

# Only used when MODE == "ind". Options: "text", "audio", "video"
MODALITY = "video"

SCRIPTS = {
    ("IEMOCAP", "ind"): "train_scripts/IEMOCAP/train_ind.py",
    ("IEMOCAP", "ta"): "train_scripts/IEMOCAP/train_ta.py",
    ("IEMOCAP", "multimodal"): "train_scripts/IEMOCAP/train_multimodal.py",
    ("MELD", "ind"): "train_scripts/MELD/train_ind_meld.py",
    ("MELD", "ta"): "train_scripts/MELD/train_ta_meld.py",
    ("MELD", "multimodal"): "train_scripts/MELD/train_multimodal_meld.py",
}

if __name__ == "__main__":
    key = (DATASET, MODE)

    if key not in SCRIPTS:
        raise ValueError(f"No training script for DATASET={DATASET!r}, MODE={MODE!r}. Valid options: {sorted(SCRIPTS)}")

    script = SCRIPTS[key]
    repo_root = os.path.dirname(os.path.abspath(__file__))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    log_suffix = ""
    if MODE == "ind":
        env["PAD_MODALITY"] = MODALITY
        log_suffix = f", MODALITY={MODALITY}"

    print(f"Running {script} (DATASET={DATASET}, MODE={MODE}{log_suffix})")
    subprocess.run([sys.executable, script], env=env, cwd=repo_root, check=True)