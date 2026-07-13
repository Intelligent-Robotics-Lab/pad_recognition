from torch.utils.data import DataLoader
from utils.iemocap_dataset import IEMOCAPDataset
from utils.iemocap_collate import iemocap_collate
from utils.split import create_splits


def get_iemocap_loaders(csv_path, batch_size=8):

    train_df, val_df, test_df = create_splits(csv_path)

    train_df.to_csv("data/train_split.csv", index=False)
    val_df.to_csv("data/val_split.csv", index=False)
    test_df.to_csv("data/test_split.csv", index=False)

    train_dataset = IEMOCAPDataset("data/train_split.csv")
    val_dataset = IEMOCAPDataset("data/val_split.csv")
    test_dataset = IEMOCAPDataset("data/test_split.csv")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=iemocap_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=iemocap_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=iemocap_collate
    )

    return train_loader, val_loader, test_loader