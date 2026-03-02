# PAD Multimodal Emotion Recognition

A Python-based multimodal emotion recognition system designed for socially aware human–robot interaction. The model predicts continuous **Pleasure, Arousal, and Dominance (PAD)** values for each conversational turn by integrating **text, auidio,** and **visual** cues.


## Model Overview

This system implements a modular pipeline grounded in affective computing and multimodal fusion:

1. **Feature Extraction**: Converts aw text, audio, and video into modality-specific embeddings.
2. **Encoding Layer**: Normalizes and projects each modality into a shared latent spcce.
3. **Cross-Modal Transformer**: Performs early fusion using attention across modalities.
4. **Regression Heads**: Three independent regressors output continuous PAD values in the range [-1,1].

The architecture supports continuous affect prediction and remains modular for continous improvement and flexibility.

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

## Activate
### Linux/macOS"
source venv/bin/activate
### Windows:
venv\Scripts\activate

### Upgrade pip
pip install --upgrade pip

### Install dependencies
pip install -r requirements.txt

## **Running Scripts**

## Usage

### Train Model
python train.py --model weights sent to saved_models

### Evaluate
python infer.py --returns predictions with target values

## License
(placeholder)

## Contributions
(placeholder)

## Citations
(placeholder)