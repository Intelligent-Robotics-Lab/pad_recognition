"""
Single entry point for inference. Edit DATASET and MODE below, then run:
    python infer_main.py

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
MODE = "ta"

# Only used when MODE == "ind". Options: "text", "audio", "video"
MODALITY = "video"

SCRIPTS = {
    ("IEMOCAP", "ind"): "inference_scripts/IEMOCAP/infer_ind.py",
    ("IEMOCAP", "ta"): "inference_scripts/IEMOCAP/infer_ta.py",
    ("IEMOCAP", "multimodal"): "inference_scripts/IEMOCAP/infer_multimodal.py",
    ("MELD", "ind"): "inference_scripts/MELD/infer_ind_meld.py",
    ("MELD", "ta"): "inference_scripts/MELD/infer_ta_meld.py",
    ("MELD", "multimodal"): "inference_scripts/MELD/infer_multimodal_meld.py",
}

if __name__ == "__main__":
    key = (DATASET, MODE)

    if key not in SCRIPTS:
        raise ValueError(f"No inference script for DATASET={DATASET!r}, MODE={MODE!r}. Valid options: {sorted(SCRIPTS)}")

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
