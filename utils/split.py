from sklearn.model_selection import train_test_split
import pandas as pd


def create_splits(csv_path):

    df = pd.read_csv(csv_path)

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