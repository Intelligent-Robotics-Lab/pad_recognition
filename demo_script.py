import torch
import pandas as pd
import matplotlib.pyplot as plt
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features
from features.text_features import extract_text_features
from models.emotion_model import EmotionPADModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = EmotionPADModel(
    text_input_dim=768,
    audio_input_dim=63,
    video_input_dim=42,
    d_model=512
).to(device)

# Load pretrained weights if available
model_path = "saved_models/emotionpad_trained.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model weights from {model_path}")
except FileNotFoundError:
    print(f"Pretrained model not found at {model_path}. Runing with random weights.")

model.eval() # Evaluation mode

# Path to your audio file
audio_path = '/home/carter/pad_recognition/data/MELD.Raw/train/dia0_utt0.wav'
# Path to video file
video_path = '/home/carter/pad_recognition/data/MELD.Raw/train/dia0_utt0.mp4'
# Path to the text file
meld_csv = '/home/carter/pad_recognition/data/MELD.Raw/train_sent_emo.csv'

# Extract audio features
audio_features = extract_audio_features(audio_path, target_sr=16000, max_seconds=None)
audio_features = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(device) # [1, T_audio, 63]

# Extract video features
video_features = extract_video_features(
    video_path,
    sample_fps=5,
    max_seconds=None,
    resize_dim=(256,256),
    include_deltas=True
)
video_features = torch.tensor(video_features, dtype=torch.float32).unsqueeze(0).to(device) # [1, T_video, 42]

# Extract first text utterance
df = pd.read_csv(meld_csv)
first_text = df.iloc[0]["Utterance"]
print("First text utterance:", first_text)
text_embeddings, attention_mask = extract_text_features(first_text)

# Forward pass through the model
with torch.no_grad():
    p, a, d = model(text_embeddings, audio_features, video_features)

print(f"Predicted PAD values for first utterance: P={p.item():.3f}, A={a.item():.3f}, D={d.item():.3f}")

# # Plot heatmaps (have to optinally figure out later)
# def plot_heatmap(feat, title, ylabel='Feature index'):
#     plt.figure(figsize=(12,6))
#     plt.imshow(feat.T, aspect='auto', origin='lower', cmap='viridis')
#     plt.colorbar(label="Normalized feature value")
#     plt.xlabel("Frame index")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.show()

# # Audio heatmap
# plot_heatmap(features, "Audio Features Heatmap", ylabel='Audio Feature index (63)')

# # Video heatmap
# plot_heatmap(video_features, "Video Features Heatmap", ylabel=f"Video Feature Index ({video_features.shape[1]})")

# # Optional plot for first utterance
# plot_heatmap(text_embeddings[0].cpu().numpy(), "Text Features Heatmap", ylabel='Embedding dimension (768)')