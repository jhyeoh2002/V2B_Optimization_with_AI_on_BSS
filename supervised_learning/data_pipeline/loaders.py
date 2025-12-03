import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. DATASET CLASS
# ==============================================================================

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for flattened Time-Series + Static + Battery features.
    
    This class takes a pandas DataFrame and flattens the 24-hour time series 
    columns into a single 1D feature vector per sample, enabling use with 
    MLPs or standard Feed-Forward networks.
    """

    def __init__(self, 
                 df: pd.DataFrame, 
                 static_cols: list, 
                 series_cols_list: list, 
                 battery_cols_list: list, 
                 target_col: str,
                 scalers: dict = None):
        """
        Args:
            df (pd.DataFrame): Source data.
            static_cols (list): List of column names for static features (e.g., time of day).
            series_cols_list (list of lists): Grouped time-series columns (e.g. [[temp_T1...temp_T24], [price_T1...]]).
            battery_cols_list (list of lists): Grouped battery columns.
            target_col (str): Name of the target column.
            scalers (dict): Dictionary containing 'static', 'series', 'battery' sklearn scalers.
        """
        self.df = df
        self.static_cols = static_cols
        self.series_cols_list = series_cols_list
        self.battery_cols_list = battery_cols_list
        self.target_col = target_col
        
        # Unpack scalers if provided
        self.static_scaler = scalers.get('static') if scalers else None
        self.series_scaler = scalers.get('series') if scalers else None
        self.battery_scaler = scalers.get('battery') if scalers else None

        # Sanity Checks
        self._validate_columns()

    def _validate_columns(self):
        """Ensures all expected columns exist in the dataframe."""
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")
        
        # Check a sample of series columns
        if self.series_cols_list and self.series_cols_list[0][0] not in self.df.columns:
             raise ValueError(f"Series column '{self.series_cols_list[0][0]}' not found.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            x (torch.Tensor): Flattened feature vector.
            y (torch.Tensor): Scalar target.
        """
        # Use iloc for speed, fallback to loc if needed, but here we expect a standardized DF
        row = self.df.iloc[idx]

        # 1. Process Static Features
        static_feats = row[self.static_cols].values.astype(np.float32)
        if self.static_scaler:
            static_feats = self.static_scaler.transform(static_feats.reshape(1, -1))[0]

        # 2. Process Time-Series Features (Flattened)
        # We concatenate [Series1_T1...T24, Series2_T1...T24] into one long vector
        series_feats = np.concatenate([
            row[cols].values.astype(np.float32) for cols in self.series_cols_list
        ])
        if self.series_scaler:
            series_feats = self.series_scaler.transform(series_feats.reshape(1, -1))[0]

        # 3. Process Battery Features
        battery_feats = np.array([], dtype=np.float32)
        if self.battery_cols_list:
            battery_feats = np.concatenate([
                row[cols].values.astype(np.float32) for cols in self.battery_cols_list
            ])
            if self.battery_scaler:
                battery_feats = self.battery_scaler.transform(battery_feats.reshape(1, -1))[0]

        # 4. Combine All -> [Static, Series, Battery]
        x = np.concatenate([static_feats, series_feats, battery_feats], axis=0)
        y = np.float32(row[self.target_col])

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ==============================================================================
# 2. SCALER MANAGEMENT
# ==============================================================================

def _manage_scalers(train_df, static_cols, series_cols_list, battery_cols_list, save_dir, fit_new=True):
    """
    Fits new scalers on training data OR loads existing scalers from disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    path_static = os.path.join(save_dir, "scaler_static.pkl")
    path_series = os.path.join(save_dir, "scaler_series.pkl")
    path_battery = os.path.join(save_dir, "scaler_battery.pkl")

    if fit_new:
        print("\t\t[INFO] Fitting new scalers on training data...")
        
        # Flatten lists for scaling
        series_cols = [c for block in series_cols_list for c in block]
        battery_cols = [c for block in battery_cols_list for c in block]

        # Fit
        s_static = StandardScaler().fit(train_df[static_cols].values)
        s_series = StandardScaler().fit(train_df[series_cols].values)
        
        if battery_cols:
            s_battery = StandardScaler().fit(train_df[battery_cols].values)
        else:
            s_battery = None

        # Save
        joblib.dump(s_static, path_static)
        joblib.dump(s_series, path_series)
        if s_battery: 
            joblib.dump(s_battery, path_battery)

        return {'static': s_static, 'series': s_series, 'battery': s_battery}

    else:
        print("\t\t[INFO] Loading existing scalers from disk...")
        if not os.path.exists(path_static):
            raise FileNotFoundError("Scaler files not found. Set fit_new=True first.")
            
        s_static = joblib.load(path_static)
        s_series = joblib.load(path_series)
        s_battery = joblib.load(path_battery) if os.path.exists(path_battery) else None
        
        return {'static': s_static, 'series': s_series, 'battery': s_battery}

# ==============================================================================
# 3. PUBLIC LOADER FUNCTION
# ==============================================================================

def get_dataloaders(
    merged_csv_path,
    feature_info_path,
    sequence_length=24,
    batch_size=32,
    num_workers=2,
    scaler_dir="scalers",
    fit_scaler=True,
    random_seed=42
):
    """
    Orchestrates the data pipeline: Load CSV -> Split -> Scale -> DataLoader.
    
    Returns:
        train_loader (DataLoader)
        val_loader (DataLoader)
    """
    # 1. Load Data & Metadata
    if not os.path.exists(merged_csv_path):
        raise FileNotFoundError(f"Dataset not found: {merged_csv_path}")
        
    df = pd.read_csv(merged_csv_path)
    
    with open(feature_info_path, "r") as f:
        info = json.load(f)

    # Extract column groups
    static_cols = info["static_cols"]
    series_cols = info["series_blocks"]
    battery_cols = info.get("battery_blocks", [])
    target_col = info["target_col"]

    # 2. Train/Val Split
    train_df, val_df = train_test_split(
        df, test_size=0.3, random_state=random_seed, shuffle=True
    )
    print(f"\t\t[INFO] Data loaded. Train: {len(train_df)}, Val: {len(val_df)}")

    # 3. Handle Scalers
    scalers = _manage_scalers(
        train_df, static_cols, series_cols, battery_cols, 
        save_dir=scaler_dir, fit_new=fit_scaler
    )

    # 4. Create Datasets
    train_ds = TimeSeriesDataset(train_df, static_cols, series_cols, battery_cols, target_col, scalers)
    val_ds = TimeSeriesDataset(val_df, static_cols, series_cols, battery_cols, target_col, scalers)

    # 5. Create Loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader