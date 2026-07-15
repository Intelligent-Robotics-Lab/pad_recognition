from utils.iemocap_dataset import IEMOCAPDataset
from features.video_features import extract_video_features


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

print("Feature shape:")
print(features.shape)

print("dtype:")
print(features.dtype)