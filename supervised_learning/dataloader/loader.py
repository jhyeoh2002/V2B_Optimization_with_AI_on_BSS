import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class TimeSeriesDataset(Dataset):
    def __init__(self, df, static_cols, series_cols_list, battery_cols_list, target_col,
                 static_scaler=None, series_scaler=None, battery_scaler=None, target_scaler=None,
                 sequence_length=24):
        """
        df: pandas DataFrame containing all features and target.
        *_scaler: fitted sklearn scalers (can be None if already normalized).
        """
        self.df = df
        self.static_cols = static_cols
        self.series_cols_list = series_cols_list
        self.battery_cols_list = battery_cols_list
        self.target_col = target_col
        self.sequence_length = sequence_length

        self.static_scaler = static_scaler
        self.series_scaler = series_scaler
        self.battery_scaler = battery_scaler
        self.target_scaler = target_scaler

        # Sanity checks
        for s_cols in series_cols_list:
            assert len(s_cols) == sequence_length, f"Series block length mismatch ({len(s_cols)} vs {sequence_length})"
        for b_cols in battery_cols_list:
            assert len(b_cols) > 0, "Empty battery block found in feature_info."

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Static features ---
        static_feats = row[self.static_cols].values.astype(np.float32)
        if self.static_scaler is not None:
            static_feats = self.static_scaler.transform(static_feats.reshape(1, -1))[0]

        # --- Time-series features (flattened) ---
        series_feats = np.concatenate([row[s_cols].values.astype(np.float32) for s_cols in self.series_cols_list])
        if self.series_scaler is not None:
            series_feats = self.series_scaler.transform(series_feats.reshape(1, -1))[0]

        # --- Battery features (flattened per vehicle) ---
        battery_feats = np.concatenate([row[b_cols].values.astype(np.float32) for b_cols in self.battery_cols_list])
        if self.battery_scaler is not None:
            battery_feats = self.battery_scaler.transform(battery_feats.reshape(1, -1))[0]

        # --- Combine all ---
        x = np.concatenate([static_feats, series_feats, battery_feats], axis=0)

        # --- Target ---
        y = np.float32(row[self.target_col])
        if self.target_scaler is not None:
            y = self.target_scaler.transform([[y]])[0, 0]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def fit_and_transform_scalers(train_df, static_cols, series_cols_list, battery_cols_list, target_col, save_dir="."):
    """Fits scalers on the training set and applies transformations to all sets."""
    os.makedirs(save_dir, exist_ok=True)

    # Flatten lists
    series_cols = [col for block in series_cols_list for col in block]
    battery_cols = [col for block in battery_cols_list for col in block]

    static_scaler = StandardScaler().fit(train_df[static_cols].to_numpy())
    series_scaler = StandardScaler().fit(train_df[series_cols].to_numpy())
    battery_scaler = StandardScaler().fit(train_df[battery_cols].to_numpy())
    target_scaler = StandardScaler().fit(train_df[[target_col]].to_numpy())

    # Save scalers for reuse
    joblib.dump(static_scaler, os.path.join(save_dir, "scaler_static.pkl"))
    joblib.dump(series_scaler, os.path.join(save_dir, "scaler_series.pkl"))
    joblib.dump(battery_scaler, os.path.join(save_dir, "scaler_battery.pkl"))
    joblib.dump(target_scaler, os.path.join(save_dir, "scaler_target.pkl"))

    return static_scaler, series_scaler, battery_scaler, target_scaler


def get_loaders_from_files(
    merged_csv_path,
    feature_info_path,
    sequence_length=24,
    batch_size=32,
    num_workers=4,
    random_seed=42,
    scaler_dir="scalers",
    fit_scaler=True,
):
    """
    Loads merged CSV and feature_infoV2.json, fits/loads scalers, splits into train/val/test,
    normalizes, and returns DataLoaders.
    """

    # --- Load dataset and feature info ---
    df = pd.read_csv(merged_csv_path)
    with open(feature_info_path, "r") as f:
        feature_info = json.load(f)

    static_cols = feature_info["static_cols"]
    series_cols_list = feature_info["series_blocks"]
    battery_cols_list = feature_info.get("battery_blocks", [])
    target_col = feature_info["target_col"]

    # --- Split data ---
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=random_seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_seed)

    print(f"✅ Dataset split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # --- Fit or load scalers ---
    if fit_scaler:
        static_scaler, series_scaler, battery_scaler, target_scaler = fit_and_transform_scalers(
            train_df, static_cols, series_cols_list, battery_cols_list, target_col, save_dir=scaler_dir
        )
        print("✅ Scalers fitted and saved.")
    else:
        static_scaler = joblib.load(os.path.join(scaler_dir, "scaler_static.pkl"))
        series_scaler = joblib.load(os.path.join(scaler_dir, "scaler_series.pkl"))
        battery_scaler = joblib.load(os.path.join(scaler_dir, "scaler_battery.pkl"))
        target_scaler = joblib.load(os.path.join(scaler_dir, "scaler_target.pkl"))
        print("✅ Scalers loaded from disk.")

    # --- Datasets ---
    train_ds = TimeSeriesDataset(train_df, static_cols, series_cols_list, battery_cols_list, target_col,
                                 static_scaler=static_scaler, series_scaler=series_scaler, battery_scaler=battery_scaler, target_scaler=None, sequence_length=sequence_length)
    val_ds = TimeSeriesDataset(val_df, static_cols, series_cols_list, battery_cols_list, target_col,
                               static_scaler=static_scaler, series_scaler=series_scaler, battery_scaler=battery_scaler, target_scaler=None, sequence_length=sequence_length)
    test_ds = TimeSeriesDataset(test_df, static_cols, series_cols_list, battery_cols_list, target_col,
                                static_scaler=static_scaler, series_scaler=series_scaler, battery_scaler=battery_scaler, target_scaler=None, sequence_length=sequence_length)

    # --- Dataloaders ---
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader = get_loaders_from_files(
        merged_csv_path="merged_windowed_datasetV2.csv",
        feature_info_path="feature_infoV2.json",
        sequence_length=24,
        batch_size=32,
        scaler_dir="scalers",
        fit_scaler=True,  # set to False when reloading later
    )
