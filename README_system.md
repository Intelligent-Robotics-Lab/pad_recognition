# PAD Multimodal Emotion Recognition

##  **1. Overview**
This is a python-based implementation of a multimodal emotional recognition system for social human-robot interaction. The system will predict emotional state from a converstaional turn using the **PAD (Pleasure, Arousal, Dominance)** model. Verbal content, vocal information, and facial expressions are processed independently, encoded into a shared space, fused through attention, and regressed into continuous emotion dimensions.

## **2. PAD Model**
The PAD framework represents emotion along three continuous psychological dimensions:

**- Pleasure (P)** - positive vs. negative affect
**- Arousal (A)** - energy vs. calmness
**- Dominance (D)** - control vs. submission

The model outputs continuous values in the range **[-1,1]**, suitable for downstream affect-aware decision making.

## **3 .System Architecture**
The system follows a modular, extensible pipeline commonly used in multimodal afffective computing that is organized into four major components:
### 3.1 **Feature Extraction**
Raw text, audio, and video inputs are converted into structured feature sequences:
- Text -> BERT token embeddings
- Audio -> MFCCs, deltas, pitch, energy, spectral features
- Video -> facial landmarks, AU-style features, head pose

All features are precomputed for efficiency and stored as tensors.

### 3.2 **Modality Encoders**
Each modality is passed through an independent encoder that normalizes and projects features into a shared embedding space.
- **Text Encoder:** A small transformer encoder processes BERT token embeddings and applies attention pooling.
- **Audio Encoder:** A LSTM-based projection encoder that captures temporal prosodic factors 
- **Video Encoder:** A LSTM-based projection encoder that captures temporal facial and nonverbal cues.
Each encoder outputs a 512-dimension embedding representing its modality.
### 3.3 **Cross-Modal Fusion**
Encoded features are fed into a **cross-modal transformer** block that performs early fusion via attention:
- Identifies complementary signals across modalities
- Downweights noisy or missing modalities
- Produces a single fused embedding representing the joint emotional signal
4. **Regression Heads**
Three independent MLP regressors map the fused embedding to continuous predictions for pleasure, arousal, and dominance. Each regressor is designed for its specific PAD dimension to accommodate differing statistical distributions.

## **4.Feature Extraction**
Feature extraction is implemented in dedicated modules under the features/ directory. 

### **4.1 Text Features**
Text features are extracted using a **BERT-based embedding pipeline:**
- Token-level embeddings are extracted and normalized per utterance
- Sequence of token embeddings is preserved for temporal information
- Resulting tensor shape: [batch, num_tokens, 768]

### **4.2 Audio Features**
Audio features capture prosodic and spectral cues associated with emotional expression. After loading and optionally downsampling the waveform, the system extracts:
- Mel-frequemcy cepstral coefficients (MFCCs)
- MFCC deltas
- Pitch
- Energy
- Spectral centroid
All features are extracted at frame level and optionally downsampled for efficiency. The resulting tensor shape is as follows: [batch, num_frames, 63]

### **4.3 Video Features**
Video features capture facial expression and nonverbal cues. Each video is sampled at 5 frames per second representing a new reading every 200ms. For each sampled frame, the system extracts:
- **Facial Action Unit Approximations** and landmarks
- **Head pose** (tilt, roll, yaw)
- **Gaze direction** and related cues
The resulting tensor shape isL [batch, num_frames, 42]

## **5. Feature Encoding**
Each modality's raw feature vector is passed through a dedicated encoder. This includes a small transformer model detected for text, and a seperate LSTM model for each the audio and video features to learn representation across time. These encoders also serve to project the heterogenous input features into a shared **512-dimensional space**. Noise injection is also utilized during training to encourage robustness.

## **6. Cross-Modal Fusion**
Early fusion is implemented using a transformer block that receives the three encoded modality ebeddings as input tokens. Cross-modal attention enables the model to:
- Identify complementary information across modalities
- Down-weight noisy or missing modalities
- Produce a single fused embedding capturing the joint emotional signal
The **512-dimensional fused embedding** output is used for all downstream regression.

## **7. Regression Heads**
Three independent regression heads predict pleasure, arousal, and dominance. Each head is a small MLP consisting of:
- Linear -> GELU -> Dropout -> Linear
Seperate regressors are used because each PAD dimension represents a distinct psychological axis with different statistical properties. These blocks predict continuous PAD values in the range [-1,1]

## **8. Peformance and Latency**
(Placeholder for evaluation metrics, inference speed, and hardware considerations.)

## **9. Limitations and Future Improvements**
(This block needs to be updated based on some of the improvements already made)
Current limitations include:
- Reliance on averaged features, which may lose fine-grained temporal dynamics
- Sensitivity to low-quality audio or video inputs
- Limited modeling of conversational context beyond single turns
Potentional Improvements:
- Incorporating temporal models (e.g., LSTMs, temporal transformers)
- Adding contextual dialogue history

## **10. Future Work**
Future extensions may explore how emotional dimensions interact over time, how dominance influences conversational flow, and how multimodal cues can be leveraged for adaptive human-robot interaction strategies.