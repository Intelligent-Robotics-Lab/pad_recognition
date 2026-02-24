import torch
from torch.utils.data import DataLoader
from utils.helpers import MELDMultimodalDataset
from models.emotion_model import EmotionPADModel
from features.text_features import prepare_text_features
from features.audio_features import extract_audio_features
from features.video_features import extract_video_features

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_dataset = MELDMultimodalDataset(root_dir=r"C:\Users\carte\emotion-recognition-hri\data\MELD.Raw", split="test")
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Initialize model
model = EmotionPADModel(text_input_dim=776, audio_input_dim=63, video_input_dim=1280, d_model=512).to(device)

# Load the trained weights
model.load_state_dict(torch.load("saved_models/emotionpad_trained.pth", map_location=device))
model.eval()  # Set the model to eval mode for inference, won't update weights

# Iterate over test batches
for text_batch, audio_paths, video_paths, pad_targets in test_loader:
    # Convert raw features to embeddings
    text_embs = torch.stack([
        torch.tensor(prepare_text_features(t), dtype=torch.float32, device=device).detach().clone()
        for t in text_batch
    ])
    audio_embs = torch.stack([
        torch.tensor(extract_audio_features(a), dtype=torch.float32, device=device).detach().clone()    
        for a in audio_paths
    ])
    video_embs = torch.stack([
        torch.tensor(extract_video_features(v), dtype=torch.float32, device=device).detach().clone()
        for v in video_paths
    ])

    # Convert target PAD values to tensor
    pad_targets_tensor = torch.stack([torch.tensor(t, dtype=torch.float32) for t in pad_targets]).to(device)

    # Forward pass
    with torch.no_grad(): # no need to compute gradients during inference
        pleasure, arousal, dominance = model(text_embs, audio_embs, video_embs)
        preds = torch.stack([pleasure, arousal, dominance], dim=1).squeeze(-1)  # shape: [batch, 3]

    # Print predictions
    for i in range(len(text_batch)):
        pred_p, pred_a, pred_d = preds[i].tolist()  # safer, gets values as floats
        target_p, target_a, target_d = pad_targets_tensor[i].tolist()  # assuming pad_targets_tensor[i] is list/tuple

        print(f"Sample {i+1}:")
        print(f"  Predicted -> P: {pred_p:.3f}, A: {pred_a:.3f}, D: {pred_d:.3f}")
        print(f"  Target    -> P: {target_p:.3f}, A: {target_a:.3f}, D: {target_d:.3f}")
        print("-" * 40)
    
    break  # remove break to run through entire test set