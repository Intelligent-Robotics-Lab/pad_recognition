# PAD Multimodal Emotion Recognition — System Documentation

## 1. Overview
A Python-based multimodal emotion recognition system for social human-robot interaction.
Predicts emotional state for a conversational turn using the **PAD (Pleasure, Arousal,
Dominance)** model. Verbal content, vocal information, and facial expressions are processed
independently, encoded into a shared space, fused through cross-modal attention, and
regressed into continuous emotion dimensions.

## 2. PAD Model
The PAD framework represents emotion along three continuous psychological dimensions:
- **Pleasure (P)** — positive vs. negative affect
- **Arousal (A)** — energy vs. calmness
- **Dominance (D)** — control vs. submission

The model outputs continuous values in `[-1, 1]` per dimension.

## 3. System Architecture
Four major components, in order: feature extraction -> modality encoders -> cross-modal
fusion -> PAD regression heads.

## 4. Feature Extraction
Implemented in `features/`. Features are extracted **on the fly, per batch**, during both
training and inference — not precomputed and cached to disk ahead of time. This is a real
cost: every epoch re-runs feature extraction for every sample.

### 4.1 Text Features
RoBERTa-large token embeddings:
- Token-level embeddings extracted and preserved per utterance (sequence, not pooled, so
  temporal/positional information survives into the encoder)
- Tensor shape: `[T, 1024]` per utterance (`[B, T, 1024]` batched)

### 4.2 Audio Features
HuBERT-large (`facebook/hubert-large-ls960-ft`), via `Wav2Vec2Processor`:
- Waveform loaded, processed, and passed through the HuBERT model
- Tensor shape: `[T, 1024]`

### 4.3 Video Features
MTCNN face detection + HSEmotion (`enet_b0_8_best_afew`) embeddings:
- Frames sampled every 5th frame
- Tensor shape: `[T, 1280]`

## 5. Modality Encoders
Implemented in `models/encoders.py`. **All three modalities use the same encoder
pattern** — a transformer encoder with attention pooling:
- 2-layer, 4-head `nn.TransformerEncoder` over the time dimension
- Learned attention pooling (`Linear -> softmax` over time) collapses the sequence to a
  single vector
- Projected/normalized to a shared `d_model = 512` embedding

### 5.1 Text Transformer Encoder (`TextTransformerEncoder`)
Input `[B, T, 1024]` -> transformer -> attention pooling -> `Linear+LayerNorm` projection
-> `[B, 512]`.

### 5.2 Audio Transformer Encoder (`AudioProjectionEncoder`)
Input `[B, T, 1024]` -> linear input projection to 512 -> transformer -> attention pooling
-> `[B, 512]`.

### 5.3 Video Transformer Encoder (`VideoProjectionEncoder`)
Input `[B, T, 1280]` -> linear input projection to 512 -> transformer -> attention pooling
-> `[B, 512]`. Structurally identical to the audio encoder aside from input dimensionality.

## 6. Cross-Modal Fusion
Implemented in `models/fusion/`. The three 512-dim embeddings are stacked to `[B, 3, 512]`
and fused via one of two selectable mechanisms (`fusion_type`):
- **`CrossModalTransformer`** (default) — a 1-layer, 1-head transformer over the 3 modality
  tokens; identifies complementary/redundant signal across modalities and produces a single
  fused `[B, 512]` embedding.
- **`MLPFusion`** — concatenates the three embeddings and passes them through an MLP.
  Simpler, no cross-modal attention.

## 7. Regression Heads
Implemented in `models/pad_regressor.py` (`PADRegressors`). A **shared** trunk feeds three
**independent** output heads:
- Shared trunk: `Linear(512->256) -> GELU -> Dropout(0.2) -> Linear(256->256) -> GELU`
- Three separate `Linear(256->1)` heads (pleasure, arousal, dominance), each passed through
  `tanh` to bound output to `[-1, 1]`

Separate heads (sharing only the trunk) because each PAD dimension is a distinct
psychological axis with different statistical properties.

## 8. Training
Two-stage pipeline, not end-to-end from scratch:

1. **Unimodal pretraining** (`train_scripts/IEMOCAP/train_ind.py`) — one encoder +
   `PADRegressors` trained alone per modality (set `MODALITY` at the top of the file, run
   once per modality). Saves fold-specific checkpoints:
   `saved_models/best_{modality}_loso_fold{N}.pth`.
2. **Fusion training** (`train_ta.py` for text+audio, `train_multimodal.py` for
   text+audio+video) — loads pretrained encoder weights, **freezes them**, and trains only
   the fusion module + regression heads. Currently loads a single fixed checkpoint
   (`saved_models/best_{modality}_model_raw.pth`) for every fold rather than the
   fold-specific one `train_ind.py` now produces — see `docs/project_state.md` for why
   that's flagged as a possible LOSO leakage risk, not yet resolved.

**Evaluation protocol:** Leave-One-Subject-Out (LOSO) cross-validation
(`utils/split.py`) — one IEMOCAP session (1-5) held out entirely as test per fold, remaining
sessions split 90/10 train/val (seed 42). All train/inference scripts loop folds 1-5 and
report per-fold and mean CCC.

**Hyperparameters:** 50 epochs, Adam, lr=1e-4, `SmoothL1Loss`, gradient clipping (norm and
value, both at 1.0), early stopping with patience=5 on validation CCC, seed=42 throughout.

## 9. Performance and Latency
No results have been logged yet under the current LOSO protocol for any pipeline; results
below predate the LOSO switch (simple train/val split) and are **not comparable** to future
LOSO numbers:
- Text only: 0.40 CCC
- Audio only: 0.46 CCC
- Text + Audio fusion: 0.5576 CCC
- Full text+audio+video: no result ever logged

Inference latency has not been benchmarked. Given feature extraction happens on the fly
(RoBERTa/HuBERT/MTCNN+HSEmotion all run per sample, per call) rather than from cached
features, per-sample latency is likely dominated by feature extraction rather than the
encoder/fusion/regressor forward pass — worth profiling before making latency claims.

## 10. Limitations and Future Work

Current limitations:
- Attention pooling collapses each modality to a single vector *before* fusion, discarding
  token/frame-level temporal alignment across modalities — the cross-modal transformer never
  sees fine-grained timing relationships between, e.g., a specific word and a specific facial
  expression.
- Sensitivity to low-quality audio or video inputs (feature extractors are pretrained on
  clean data).
- No modeling of conversational context beyond a single turn.
- Feature extraction is not cached, so training/inference cost scales with dataset size on
  every pass rather than being paid once.
- MELD support exists as scaffolding but is not currently functional (model API mismatches,
  incompatible video feature dimensionality vs. IEMOCAP) — single-dataset only for now.
- The fusion-stage frozen-encoder / LOSO-fold mismatch noted in Section 8 is an open
  methodological question, not just an implementation detail.

Potential improvements:
- Resolve the frozen-encoder/fold mismatch (load per-fold pretrained encoders in the fusion
  stage instead of a fixed checkpoint)
- Cache extracted features instead of recomputing per epoch
- Add MELD as a genuine second dataset (would require aligning video feature dimensionality
  and fixing the model API calls)
- Extend beyond single-turn context to model dialogue history
- Investigate whether attention-pooling weights are behaving as intended — there's an open,
  unresolved investigation into this in the encoder code as of 2026-07-31 (see
  `docs/project_state.md`)
