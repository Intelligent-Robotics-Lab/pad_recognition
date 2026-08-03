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
├── data/                    # IEMOCAP (working) and MELD.Raw (unused/non-functional path)
├── features/                # Per-modality feature extraction (RoBERTa / HuBERT / MTCNN+HSEmotion)
├── models/                  # Encoders, fusion modules, regression heads, assembled models
├── train_scripts/IEMOCAP/   # train_ind.py, train_ta.py, train_multimodal.py
├── inference_scripts/IEMOCAP/  # infer_ind.py, infer_ta.py, infer_multimodal.py
├── train_scripts/MELD/, inference_scripts/MELD/, utils/MELD/  # non-functional, do not use yet
├── saved_models/            # Checkpoints (see naming caveat below)
├── utils/                   # Shared helpers, including utils/split.py (LOSO logic)
├── docs/                    # Project state + decision log (living docs, start here)
└── requirements.txt         # Known stale — see Setup below
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

**Known gap:** `requirements.txt` does not list `facenet_pytorch` or `hsemotion`, both of
which `features/video_features.py` imports directly — a clean install will fail at video
feature extraction time until this is corrected.

## Running Scripts

There's no CLI — scripts are run directly, and modality/fusion choices are set by editing a
variable near the top of the file, not via arguments.

### Train
Each script internally loops all 5 LOSO folds (one IEMOCAP session held out as test per
fold) and prints per-fold and mean validation CCC.

- `python train_scripts/IEMOCAP/train_ind.py` — trains one unimodal encoder + regressor.
  Set `MODALITY = "text" | "audio" | "video"` at the top of the file. Run once per
  modality. Saves `saved_models/best_{modality}_loso_fold{N}.pth` per fold.
- `python train_scripts/IEMOCAP/train_ta.py` — text+audio fusion. Loads frozen pretrained
  text/audio encoders from `saved_models/best_text_model_raw.pth` /
  `best_audio_model_raw.pth` (fixed checkpoints, not fold-specific — see caveat below).
- `python train_scripts/IEMOCAP/train_multimodal.py` — full text+audio+video fusion. Same
  frozen-encoder pattern, also loads `best_video_model_raw.pth`. Set
  `FUSION_TYPE = "transformer" | "mlp"` at the top.

**Checkpoint-naming caveat:** `train_ind.py` now saves fold-specific checkpoints
(`best_text_loso_fold1.pth`, etc.), but `train_ta.py`/`train_multimodal.py` still load a
single fixed `_raw` checkpoint for every fold. That means the frozen encoder used while a
given session is held out as test may have already seen that session during its own
(pre-LOSO) pretraining — a possible train/test leak in the fusion stage. See
`docs/project_state.md` for details; this hasn't been resolved yet.

### Inference
Mirrors the training scripts one-to-one (same LOSO fold loop, same `MODALITY`/`FUSION_TYPE`
variables to edit): `infer_ind.py`, `infer_ta.py`, `infer_multimodal.py` in
`inference_scripts/IEMOCAP/`. Computes RMSE, MAE, Pearson r, and CCC per PAD dimension.
Checkpoints are pulled from `saved_models/` — since scripts overwrite "best" checkpoints on
every improving run, save off any checkpoint you want to keep before retraining.

### MELD
`train_scripts/MELD/`, `inference_scripts/MELD/`, and `utils/MELD/` exist but are not
currently functional against the present model API (constructor/output-shape mismatches,
and MELD's video features are a different dimensionality than IEMOCAP's). Don't use this
path yet — see `docs/decision_log.md` for what would need to change.

## Results
No results have been logged yet under the current LOSO protocol for any pipeline. Earlier
CCC numbers from this project (pre-LOSO, simple train/val split) were: Text 0.40, Audio
0.46, Text+Audio fusion 0.5576. These are not comparable to future LOSO numbers and are
kept only in `docs/decision_log.md` for historical reference.

## Future Work
(placeholder)

## License
(placeholder)

## Contributions
(placeholder)

## Citations
(placeholder)
