import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

class TimeSeriesDataset(Dataset):
    def __init__(self, df, static_cols, series_cols_list, target_col, sequence_length=24):
        """
        df: pandas DataFrame containing all features and target.
        static_cols: list of names of static-feature columns.
        series_cols_list: list of lists of column names, one per time-series block.
        target_col: name of target column.
        sequence_length: number of time steps per series (default=24).
        """
        self.df = df
        self.static_cols = static_cols
        self.series_cols_list = series_cols_list
        self.target_col = target_col
        self.sequence_length = sequence_length

        # sanity check
        for s_cols in series_cols_list:
            assert len(s_cols) == sequence_length, f"Series block length mismatch ({len(s_cols)} vs {sequence_length})"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # static features
        static_feats = row[self.static_cols].values.astype(np.float32)

        # time-series features (flattened)
        series_feats = []
        for s_cols in self.series_cols_list:
            series_feats.append(row[s_cols].values.astype(np.float32))
        series_feats = np.concatenate(series_feats, axis=0)  # shape: (#series * seq_len,)

        # Combine all features
        x = np.concatenate([static_feats, series_feats], axis=0)

        # target
        y = np.float32(row[self.target_col])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_loaders_from_files(
    merged_csv_path,
    feature_info_path,
    sequence_length=24,
    batch_size=32,
    num_workers=4,
    random_seed=42,
):
    """
    Loads merged CSV and feature_info.json, splits data into train/val/test (70/15/15),
    and returns PyTorch DataLoaders.
    """

    # --- load data ---
    df = pd.read_csv(merged_csv_path)
    with open(feature_info_path, "r") as f:
        feature_info = json.load(f)

    static_cols = feature_info["static_cols"]
    series_cols_list = feature_info["series_blocks"]
    target_col = feature_info["target_col"]

    # --- random split (reproducible) ---
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=random_seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_seed)

    print(f"✅ Dataset split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # --- datasets ---
    train_ds = TimeSeriesDataset(train_df, static_cols, series_cols_list, target_col, sequence_length)
    val_ds   = TimeSeriesDataset(val_df,  static_cols, series_cols_list, target_col, sequence_length)
    test_ds  = TimeSeriesDataset(test_df, static_cols, series_cols_list, target_col, sequence_length)

    # --- dataloaders ---
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
