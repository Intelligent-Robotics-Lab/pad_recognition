from torch.utils.data import DataLoader
from utils.iemocap_dataset import IEMOCAPDataset
from utils.iemocap_collate import iemocap_collate
from utils.split import create_splits
from utils.meld_dataset import MELDDataset
from utils.meld_collate import meld_collate


def get_iemocap_loaders(csv_path, batch_size=8, split="loso", fold=1):

    train_df, val_df, test_df = create_splits(csv_path, split=split, fold=fold)

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


def get_meld_loaders(root_dir="data/MELD.Raw", batch_size=1):
    # No need for LOSO becuase MELD already has its own splits

    train_dataset = MELDDataset(root_dir, split="train")
    val_dataset = MELDDataset(root_dir, split="dev")
    test_dataset = MELDDataset(root_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=meld_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=meld_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=meld_collate
    )

    return train_loader, val_loader, test_loader