import torch
from utils.iemocap_dataset import IEMOCAPDataset
from features.video_features import extract_video_features
from models.encoders import VideoProjectionEncoder


ds = IEMOCAPDataset("data/iemocap.csv")

sample = ds[0]

print("Video:")
print(sample["video_path"])

print("Time:")
print(sample["start_time"], sample["end_time"])


features = extract_video_features(
    sample["video_path"],
    sample["start_time"],
    sample["end_time"]
)

features = torch.tensor(features).unsqueeze(0)

print(features.shape)

encoder  = VideoProjectionEncoder()

out = encoder(features)

print(out.shape)

print("Feature shape:")
print(features.shape)

print("dtype:")
print(features.dtype)