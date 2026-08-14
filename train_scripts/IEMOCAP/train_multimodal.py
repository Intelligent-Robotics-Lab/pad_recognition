"""
Train script to test text, audio, and video modalaties together using the IEMOCAP dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from features.text_features import extract_text_features
from features.audio_features import extract_audio_features

from models.emotion_model import EmotionPADModel

from utils.dataloaders import get_iemocap_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Valid options: "mlp" and "transformer"
FUSION_TYPE = os.environ.get("PAD_FUSION_TYPE", "transformer")

num_epochs = 50
learning_rate = 1e-4
seed = 42

# Still initialize despite not using random split as it "could" still affect parameters
torch.manual_seed(seed)

# CCCs for evaluation only
@torch.no_grad()
def ccc_score(pred, target):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)

    pred_var = pred.var(dim=0, unbiased=False) + 1e-6
    target_var = target.var(dim=0, unbiased=False) + 1e-6

    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)

    ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + 1e-8)
    return ccc, ccc.mean().item()


# Evaluate the model on an entire dataset split (using CCCs)
def evaluate(model, loader, name="VAL"):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            text = batch["text"][0]
            audio = batch["audio"][0]
            sample_rate = batch["sample_rate"][0]

            target = batch["pad"].to(device)

            # Compute pretrained text and audio features
            text_feats = extract_text_features(text)
            audio_feats = extract_audio_features(audio, sample_rate)
            video_feats = batch["video_feats"][0]

            # Convert features to tensors on the right device
            text_feats = torch.as_tensor(text_feats, dtype=torch.float32, device=device)
            audio_feats = torch.as_tensor(audio_feats, dtype=torch.float32, device=device)
            video_feats = torch.as_tensor(video_feats, dtype=torch.float32, device=device)

            # Ensure proper dimensionality
            if text_feats.dim() == 2:
                text_feats = text_feats.unsqueeze(0)

            if audio_feats.dim() == 2:
                audio_feats = audio_feats.unsqueeze(0)

            if video_feats.dim() == 2:
                video_feats = video_feats.unsqueeze(0)

            # Forward pass
            pred = model(text_feats, audio_feats, video_feats)

            preds_all.append(pred)
            targets_all.append(target)

    # Store predictions and perform CCC calculation
    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    ccc, avg = ccc_score(preds_all, targets_all)

    print(
        f"{name} CCC | "
        f"P: {ccc[0]:.4f}  "
        f"A: {ccc[1]:.4f}  "
        f"D: {ccc[2]:.4f}  "
        f"Avg: {avg:.4f}"
    )

    model.train()
    return avg

# Training loop
def train_fold(fold):

    print(f"\nTraining Multimodal ({FUSION_TYPE}) | Fold {fold}")

    # Model initialize (Text-audio model)
    model = EmotionPADModel(text_hidden_dim=1024, audio_input_dim=1024, video_input_dim=1280, d_model=512, fusion_type=FUSION_TYPE).to(device)

    # Load pretrained weights from this fold's unimodal encoders (train_ind.py file)
    # Fold-specific so the encoder never saw this fold's held-out session during pretraining
    text_checkpoint = torch.load(f"saved_models/best_text_loso_fold{fold}.pth", map_location=device)
    audio_checkpoint = torch.load(f"saved_models/best_audio_loso_fold{fold}.pth", map_location=device)
    video_checkpoint = torch.load(f"saved_models/best_video_loso_fold{fold}.pth", map_location=device)

    # Remove the "encoder." prefix so the weights match EmotionPADModelTA declared above
    text_encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in text_checkpoint.items()
        if k.startswith("encoder.")
    }

    audio_encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in audio_checkpoint.items()
        if k.startswith("encoder.")
    }

    video_encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in video_checkpoint.items()
        if k.startswith("encoder.")
    }

    # Initialize the model encoders with the pretrained weights
    model.text_encoder.load_state_dict(text_encoder_state)
    model.audio_encoder.load_state_dict(audio_encoder_state)
    model.video_encoder.load_state_dict(video_encoder_state)

    print("Loaded pretrained encoders.")

    # Freeze the encoders so only the fusion module and regressor are trained
    for param in model.text_encoder.parameters():
        param.requires_grad = False

    for param in model.audio_encoder.parameters():
        param.requires_grad = False

    for param in model.video_encoder.parameters():
        param.requires_grad = False

    # Only optimize trainable parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    train_loader, val_loader, test_loader = get_iemocap_loaders(
        "data/iemocap.csv",
        batch_size=1,
        split="loso",
        fold=fold,
    )

    print(
        f"Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)} | "
    )

    best_val_ccc = -float("inf")
    os.makedirs("saved_models", exist_ok=True)

    # Early stopping settings
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            text = batch["text"][0]
            audio = batch["audio"][0]
            sample_rate = batch["sample_rate"][0]

            target = batch["pad"].to(device)

            text_feats = extract_text_features(text)
            audio_feats = extract_audio_features(audio, sample_rate)
            video_feats = batch["video_feats"][0]

            # Convert features to tensors
            text_feats = torch.as_tensor(text_feats, dtype=torch.float32, device=device)
            audio_feats = torch.as_tensor(audio_feats, dtype=torch.float32, device=device)
            video_feats = torch.as_tensor(video_feats, dtype=torch.float32, device=device)

            if text_feats.dim() == 2:
                text_feats = text_feats.unsqueeze(0)

            if audio_feats.dim() == 2:
                audio_feats = audio_feats.unsqueeze(0)

            if video_feats.dim() == 2:
                video_feats = video_feats.unsqueeze(0)

            # Forward pass through the model 
            optimizer.zero_grad()

            pred = model(text_feats, audio_feats, video_feats)

            SmoothL1Loss = loss_fn(pred, target).mean(dim=1)
            loss = SmoothL1Loss.mean()

            # Backpropagation
            loss.backward()

            # Update model parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0,)

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Train MSE: {avg_loss:.4f}")

        # Validation with CCC loss
        val_ccc = evaluate(model, val_loader, "VAL")

        min_delta = 1e-3

        # Save the best model only
        if val_ccc > best_val_ccc + min_delta:
            best_val_ccc = val_ccc
            epochs_without_improvement = 0

            save_path = os.path.join("saved_models", f"best_tav_{FUSION_TYPE}_loso_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)

            print(
                f"Saved new best model "
                f"(VAL CCC = {best_val_ccc:.4f})"
            )
        
        else:
            epochs_without_improvement += 1
            print(f"No improvements for {epochs_without_improvement}/{patience} epochs.")

            if epochs_without_improvement >= patience:
                print("\nEarly stopping triggered.")
                break

    print(f"\nFold {fold} complete.")
    print(f"Best validation CCC: {best_val_ccc:.4f}")

    return best_val_ccc

results = {}

for fold in range(1, 6):
    results[fold] = train_fold(fold)

print("\nFinal LOSO Validation Results")

for fold, score in results.items():
    print(f"Fold {fold}: {score:.4f}")

mean_ccc = sum(results.values()) / len(results)

print(f"\nMean Validation CCC: {mean_ccc:.4f}")