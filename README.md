# PAD Multimodal Emotion Recognition

A Python-based multimodal emotion recognition system designed for socially aware human–robot interaction. The model predicts continuous **Pleasure, Arousal, and Dominance (PAD)** values for each conversational turn by integrating **text, auidio,** and **visual** cues.

This repository implements a modern, transformer-based multimodal architecture suitable for research in affective computing, social robotics, and Human-AI interaction.

# Key Features
- Multimodal PAD prediction (continuous values in [-1,1])
- Transformer-based **cross-modal attention fusion**
- Independent **text, audio, and video encoders**
- Precomputed feature pipeline for fast training
- Evaluation metrics: RMSE, MAE, Pearson, CCC

## Model Overview

This system implements a modular pipeline grounded in affective computing and multimodal fusion:

1. **Feature Extraction**
Each modality is processed into structured, frame-level or token-level features:
- **Text:** BERT token embeddings
- **Audio:** MFCCs, deltas, pitch, energy, spectral features
- **Video:** Facial landmarks, AU-style features, head pose
These features are precomputed for efficiency.
2. **Modality Encoders**
Each modality is encoded independently:
- **Text Encoder:** Transformer-based sequence encoder -> 512-dim vector
- **Audio Encoder:** LSTM over temporal acoustic features -> 512-dim vector
- **Video Encoder:** LSTM over frame-level facial features -> 512-dim vector
All encoders project into a shared latent space ('d_model = 512')
3. **Cross-Modal Transformer Fusion**
A lightweight transformer encoder performs **cross‑modal attention** over the three modality embeddings: [text, audio, video] -> fused representation.
This allows the model to learn how modalities influence each other.
4. **PAD Regression Heads**
Three independent regression heads predict:
- **Pleasure**
- **Arousal**
- **Dominance**
Outputs are continuous and constrained to [-1,1]

## Project Structure
pad-emotion/
- data/             # Raw and preprocessed data files
- features/         # Feature extraction files
- models/           # Model definitions
- saved_models/     # Saved training weights
- utils/            # Helper utilities

- train.py
- infer.py
- README_system.md  # Detailed system documentation
- README.md         # Main README with setup instructions
- requirements.txt  # Python dependencies

## Clone Repository
git clone https://github.com/Intelligent-Robotics-Lab/pad_recognition
cd pad-emotion

## **Set Up Virtual Environment and Depencies**

## Virtual Environment Setup

### Create and activate a Python virtual environment
python -m venv venv

### Activate
**Linux/macOS**
source venv/bin/activate
**Windows**
venv\Scripts\activate

### Upgrade pip
pip install --upgrade pip

### Install dependencies
pip install -r requirements.txt

## **Running Scripts**

## Usage

### Train Model
Precompute feautres first (see precompute_features.py and features/ directory), then run:
python train.py
Model checkpoints will be saved to:
saved_models/emotionpad_trained.pth

### Evaluate
run python infer.py
This computes: RMSE, MAE, Pearson Correlation, and Concordance Correlation Coefficient (CCC) for each PAD dimension.

## Example Benchmark (Placeholder)

## Future Work
Planned improvements and research directions:

## License
(placeholder)

## Contributions
(placeholder)

## Citations
(placeholder)