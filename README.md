# **PAD Multimodal Emotion Recognition**

This project presents a Python-based multimodal emotion recognition system designed for socially aware human–robot interaction. The model predicts continuous **Pleasure, Arousal, and Dominance (PAD)** values for each conversational turn by integrating **text, audio,** and **visual** cues.

This repository implements a modern, transformer-based multimodal architecture suitable for research in affective computing, social robotics, and Human-AI interaction.

## **Key Features**
- Multimodal PAD prediction (continuous values in [-1,1])
- Transformer-based **cross-modal attention fusion**
- Independent **text, audio, and video encoders**
- Precomputed feature pipeline for fast training
- Evaluation metrics: RMSE, MAE, Pearson, CCC

### **Model Overview**

This system implements a modular pipeline grounded in affective computing and multimodal fusion:

1. **Feature Extraction**
Each modality is processed into structured, frame-level or token-level features:
- **Text:** BERT token embeddings
- **Audio:** Wav2Vec feature embeddings
- **Video:** FER-style emotion prediction
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

## **Project Structure**
pad-emotion/
- data/                 # Raw and preprocessed data files
- features/             # Feature extraction files
- inference_scripts/    # Evaluation and test scripts
- models/               # Model definitions
- saved_models/         # Saved training weights
- train_scripts/        # All necessary training scripts
- utils/                # Helper utilities

- README_system.md      # Detailed system documentation
- README.md             # Main README with setup instructions
- requirements.txt      # Python dependencies for Dockerfile (future update)

## **Clone Repository**
git clone https://github.com/Intelligent-Robotics-Lab/pad_recognition
cd pad_recognition

## **Set Up Virtual Environment and Depencies**

### **Virtual Environment Setup**

#### **Create and activate a Python virtual environment**
python -m venv venv

#### **Activate**
**Linux/macOS**
source venv/bin/activate
**Windows**
venv\Scripts\activate

#### **Upgrade pip**
pip install --upgrade pip

#### **Install dependencies**
pip install -r requirements.txt
- this will be updated in future iterations to be a dockerized container

## **Running Scripts**

### **Usage**

#### **Train Model**
Precompute feautres first (see precompute_features.py and features/ directory), then run:
- python train.py for full model predictions
- python train_single_modality.py for single modality prediction (change line 49 to switch modality)
- python train_text_audio.py for fused text and audio prediction
Model checkpoints will be saved to:
saved_models/emotionpad_trained.pth

#### **Evaluation and Inference**
Testing with the metrics RMSE, MAE, Pearson Correlation, and Concordance Correlation Coefficient (CCC) for each PAD dimension can be performed for both the full model and single modality.

Use the precomputed test features as described in training and run:
- python infer.py for full model testing with full saved model weights
- python infer_single_modality.py for single modality testing (change line 16 to switch modality)
- python infer_text_audio.py for fused text and audio testing

The inference model pulls the appropriate weights from the saved_models folder after training is completed. These models will constantly be overwritten by new improvements so benchmarks need to be saved if they are to be accessed.

## **Example Benchmark**
(placeholder)

## **Future Work**
(placeholder)

## **License**
(placeholder)

## **Contributions**
(placeholder)

## **Citations**
(placeholder)