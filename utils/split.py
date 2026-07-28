import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(csv_path, split="loso", fold=1):

    df = pd.read_csv(csv_path)

    if split == "random":

        train_df, temp_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            shuffle=True
        )

        return train_df, val_df, test_df

    elif split == "loso":

        # Held out session
        test_df = df[df["session"] == fold]

        # Remaining sessions
        remaining = df[df["session"] != fold]

        train_df, val_df = train_test_split(
            remaining,
            test_size=0.10,
            random_state=42,
            shuffle=True,
        )

        return train_df, val_df, test_df

    else:
        raise ValueError(f"Unknown split type: {split}")