import torch

sample = torch.load("data/MELD.Raw/precomputed_v4/train_features_0.pt")[0]
sample1 = torch.load("data/MELD.Raw/precomputed_v4/dev_features_0.pt")[0]

print("Text shape:", sample["text"].shape)
print("Audio shape:", sample["audio"].shape)
print("Video shape:", sample["video"].shape)

print("Text shape:", sample1["text"].shape)
print("Audio shape:", sample1["audio"].shape)
print("Video shape:", sample1["video"].shape)