# PAD Multimodal Emotion Recognition

A Python-based multimodal emotion recognition system for socially aware human-robot
interaction. Predicts continuous **Pleasure, Arousal, and Dominance (PAD)** values in
`[-1, 1]` for a conversational turn by integrating text, audio, and video cues, trained on
IEMOCAP.

## Key Features
- Multimodal PAD prediction, continuous values in `[-1, 1]`
- Independent text, audio, and video encoders (all transformer-based with attention
  pooling), projecting into a shared 512-dim space
- Cross-modal transformer fusion (or a simpler MLP-concat fusion, selectable)
- Two-stage training: unimodal encoders pretrained first, then frozen while fusion +
  regression heads train
- Leave-One-Subject-Out (LOSO) cross-validation across all training/inference scripts
- Evaluation metrics: RMSE, MAE, Pearson correlation, CCC

## Model Overview

1. **Feature Extraction** (`features/`) — extracted on the fly, per batch, during
   training/inference (not precomputed and cached to disk):
   - **Text:** RoBERTa-large token embeddings, `[T, 1024]`
   - **Audio:** HuBERT-large (`facebook/hubert-large-ls960-ft`) via `Wav2Vec2Processor`,
     `[T, 1024]`
   - **Video:** MTCNN face detection + HSEmotion (`enet_b0_8_best_afew`) embeddings,
     sampled every 5th frame, `[T, 1280]`
2. **Modality Encoders** (`models/encoders.py`) — each modality gets its own
   transformer encoder (2 layers, 4 heads) with learned attention pooling over the time
   dimension, projecting to a shared `d_model = 512` space:
   - `TextTransformerEncoder`, `AudioProjectionEncoder`, `VideoProjectionEncoder`
3. **Cross-Modal Fusion** (`models/fusion/`) — the three 512-dim embeddings are stacked
   into `[B, 3, 512]` and fused via either:
   - `CrossModalTransformer` (1 layer, 1 head) — the default, or
   - `MLPFusion` (concat + MLP)
4. **PAD Regression Heads** (`models/pad_regressor.py`) — a shared MLP
   (`512 -> 256 -> 256`, GELU) feeds three independent linear heads (pleasure, arousal,
   dominance), each `tanh`-bounded to `[-1, 1]`.

Model variants (`models/`):
- `EmotionPADModel` — full text+audio+video model
- `EmotionPADModelTA` — text+audio only
- `SingleModalityModel` — single encoder + regressor, used to pretrain each modality alone

## Project Structure
```
pad_recognition/
├── train_main.py            # Single entry point for training — see Running Scripts below
├── infer_main.py            # Single entry point for inference
├── data/                    # IEMOCAP and MELD.Raw
├── features/                # Per-modality feature extraction (RoBERTa / HuBERT / MTCNN+HSEmotion)
├── models/                  # Encoders, fusion modules, regression heads, assembled models
├── train_scripts/IEMOCAP/   # train_ind.py, train_ta.py, train_multimodal.py
├── train_scripts/MELD/      # train_ind_meld.py, train_ta_meld.py, train_multimodal_meld.py
├── inference_scripts/IEMOCAP/  # infer_ind.py, infer_ta.py, infer_multimodal.py
├── inference_scripts/MELD/     # infer_ind_meld.py, infer_ta_meld.py, infer_multimodal_meld.py
├── saved_models/            # Checkpoints (best_* overwritten on every improving run)
├── utils/                   # Shared helpers: utils/split.py (LOSO logic), utils/meld_*.py,
│                             # utils/dataloaders.py, etc. — flat layout, no per-dataset subfolders
├── docs/                    # Project state + decision log (living docs, start here)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Intelligent-Robotics-Lab/pad_recognition
cd pad_recognition

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Running Scripts

The simplest way to run anything is the two entry-point scripts at the repo root —
`train_main.py` and `infer_main.py`. Edit the constants at the top of the file, then run it:

```python
DATASET = "IEMOCAP"   # "IEMOCAP" or "MELD"
MODE = "ta"            # "ind", "ta", "multimodal"
MODALITY = "video"     # only used when MODE == "ind": "text", "audio", "video"
```

```bash
python train_main.py
python infer_main.py
```

This just resolves those choices to the right underlying script and runs it with
`PYTHONPATH` set automatically — there's still no argument-based CLI, and fusion type
(`FUSION_TYPE = "transformer" | "mlp"`) is set inside the target `ta`/`multimodal` script
itself, same as before. You can also run any script directly
(`python train_scripts/IEMOCAP/train_ta.py`, etc.) exactly as before; the dispatcher is a
convenience layer, not a replacement.

### Train
Each script internally loops all 5 LOSO folds (one IEMOCAP session held out as test per
fold) and prints per-fold and mean validation CCC.

- `train_ind.py` (`MODE="ind"`) — trains one unimodal encoder + regressor per `MODALITY`.
  Saves `saved_models/best_{modality}_loso_fold{N}.pth` per fold.
- `train_ta.py` (`MODE="ta"`) — text+audio fusion. Loads the fold-matched frozen text/audio
  encoders (`best_{text,audio}_loso_fold{N}.pth`) for each fold — not a fixed checkpoint, so
  there's no train/test leak across folds.
- `train_multimodal.py` (`MODE="multimodal"`) — full text+audio+video fusion, same
  fold-matched frozen-encoder pattern including video.

### Inference
Mirrors the training scripts one-to-one (same LOSO fold loop, same fold-matched checkpoint
loading): `infer_ind.py`, `infer_ta.py`, `infer_multimodal.py` in
`inference_scripts/IEMOCAP/`. Computes RMSE, MAE, Pearson r, and CCC per PAD dimension.
Checkpoints are pulled from `saved_models/` — since scripts overwrite "best" checkpoints on
every improving run, save off any checkpoint you want to keep before retraining.

### MELD
`train_scripts/MELD/` and `inference_scripts/MELD/` mirror the IEMOCAP scripts structurally
and mechanically — same feature extractors, same training/inference logic, same LOSO-style
per-fold checkpoint naming (MELD instead uses its original fixed train/dev/test split rather
than LOSO folds — see `docs/decision_log.md`). This is prep/connective work, not a push for
real MELD results: no MELD checkpoints exist yet, since no real training run has been done.

## Results
No results have been logged yet under the current LOSO protocol for any pipeline — baseline
collection across all 5 IEMOCAP configs (text, audio, video, text+audio, full multimodal) is
in progress. Earlier CCC numbers from this project (pre-LOSO, simple train/val split) were:
Text 0.40, Audio 0.46, Text+Audio fusion 0.5576. These are not comparable to the upcoming
LOSO numbers and are kept only in `docs/decision_log.md` for historical reference. See
`docs/project_state.md` for the current baseline-collection checklist.

## Future Work
(placeholder)

## License
(placeholder)

## Contributions
(placeholder)

## Citations
(placeholder)
