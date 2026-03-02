# PAD Multimodal Emotion Recognition

##  **1. Overview**
This is a python-based implementation of a multimodal emotional recognition system for social human-robot interaction. The system will predict emotional state from a converstaional turn using the **PAD (Pleasure, Arousal, Dominance)** model. Verbal content, vocal information, and facial expressions are processed independently, encoded into a shared space, fused through attention, and regressed into continuous emotion dimensions.

## **2. PAD Model**
The PAD framework represents emotion along three continuous axes:

**- Pleasure (P)** - positive vs. negative affect
**- Arousal (A)** - activation vs. calmness
**- Dominance (D)** - control vs. submission

Each dimension is predicted as a continuous value within a configurable range, set to [-1,1] in this implementation.

## **3 .System Architecture**
The system follows a modular, extensible pipeline commonly used in multimodal afffective computing:
1. **Feature Extraction**
Raw text, audio and video inputs are transformed into moadlity-specific feature vectors.
2. **Feature Encoding**
Each modality is passed through an independent encoder that normalizes and projects features into a shared embedding space.
3. **Cross-Modal Fusion**
Encoded features are fed into a transformer blok that performs early fusion through cross-modal attention, producing a single fused embedding.
4. **Regression Heads**
Three independent MLP regressors map the fused embedding to continuous predictions for pleasure, arousal, and dominance.

The final output isa 3-dimensional vector representing the emotional state of the conversational turn.

## **4.Feature Extraction**
Feature extraction is implemented in dedicated modules under the features/ directory. The system is designed to remain robust when one or more modalities are missing or noisy. Missing modalities are replaced with zero vectors, allowing the model to learn to ignore absent signals during training

### **4.1 Text Features**
Text features capture semantic, affective, and emotional information:
- A **BERT-based encoder** generates contextual sentence embeddings by averaging token-level hidden states.
- A **sentiment classifier** produces a polarity score mapped to a signed scalar.
- An **emotion classifer** outputs a distribution over emotion emotion categories; the highest-confidence score is used as a compact emotional signal.

These components are concatenated into a single text feature vector containing:
- BERT embedding
- Sentiment score
- Emotion confidence score
This vector is passed to the text encoder.

### **4.2 Audio Features**
Audio features capture prosodic and spectral cues associated with emotional expression. After loading and optionally downsampling the waveform, the system extracts:
- Mel-frequemcy cepstral coefficients (MFCCs)
- MFCC deltas
- Pitch
- Energy
- Spectral centroid
All features are averaged across time and concatenated into a single audio feature vector.

### **4.3 Video Features**
Video features capture facial expression and nonverbal cues. Each video is sampled at a reduced frame rate to balance computational cost and temporal coverage.

For each sampled frame, the system extracts:
- **Facial Action Units (FAUs)** (18 dimensions)
- **Head pose** (tilt, roll, yaw)
- **Gaze direction** and related cues
Frame-level features are averaged to produce a single video feature vector representing the entire utterance.

# **5. Feature Encoding**
Each modality's raw feature vector is passed through a dedicated encoder consiting of:
- Linear projection
- GELU activation
- Dropout
- Second linear layer
- Layer normalization
The encoders map heterogenous feature spaces into a unified embedding dimension suitable for multimodal fusion.

## **6. Cross-Modal Fusion**
Early fusion is implemented using a transformer block that receives the three encoded modality ebeddings as input tokens. Cross-modal attention enables the model to:
- Identify complementary information across modalities
- Down-weight noisy or missing modalities
- Produce a single fused embedding capturing the joint emotional signal
This fused representation is used for all downstream predictions.

## **7. Regression Heads**
Three independent regression heads predict pleasure, arousal, and dominance. Each head is a small MLP consisting of:
- Linear -> GELU -> Dropout -> Linear
Seperate regressors are used because each PAD dimension represents a distinct psychological axis with different statistical properties.

## **8. Peformance and Latency**
(Plceholder for evaluation metrics, inference speed, and hardware considerations.)

## **9. Limitations and Future Improvements**
Current limitations include:
- Reliance on averaged features, which may lose fine-grained temporal dynamics
- Sensitivity to low-quality audio or video inputs
- Limited modeling of conversational context beyond single turns
Potentional Improvements:
- Incorporating temporal models (e.g., LSTMs, temporal transformers)
- Adding contextual dialogue history

## **10. Future Work**
Future extensions may explore how emotional dimensions interact over time, how dominance influences conversational flow, and how multimodal cues can be leveraged for adaptive human-robot interaction strategies.