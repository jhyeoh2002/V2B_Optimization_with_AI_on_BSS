import os
from sys import prefix
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.utils.battscheduler import schedule_batteries
# import time_series_generator.time_series_generator.core as tsg


class BatterySeriesGenerator:
    """
    A class to generate, augment, and schedule battery time series data.

    Parameters
    ----------
    data_path : str
        Path to the input CSV of fully charged battery data.
    processed_dir : str
        Directory where processed .npy files will be saved.
    """

    def __init__(self, 
                 data_path='../time_series_generator/modified_data/resample_data.csv', 
                 processed_dir='./processed'):
        self.data_path = data_path
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Generate or load series
    # -------------------------------------------------------------------------
    def generate_battery_series(
        self,
        window_size: int = 48,
        tolerance: int = 6,
        n_samples: int = 25,
        rerun: bool = False
    ):
        """
        Crash-safe generation of battery series with resumable checkpoints.

        Parameters
        ----------
        window_size : int
            Length of each time window.
        tolerance : int
            Allowed NaN count for artificial generation.
        n_samples : int
            Number of artificial samples per eligible chunk.
        rerun : bool
            If True, ignore existing outputs and checkpoint; regenerate from scratch.
        """

        # === Setup paths ===
        path_clean = os.path.join(self.processed_dir, f"battery_series_window{window_size}.npy")
        path_nan   = os.path.join(self.processed_dir, f"battery_series_with_{tolerance}nan_window{window_size}.npy")
        # Use base name for checkpoint, extension will be added later
        ckpt_base = os.path.join(self.processed_dir, f"battery_series_ckpt_{tolerance}nan_window{window_size}")
        ckpt_path = ckpt_base + ".npz"
        tmp_path  = ckpt_base + ".tmp.npz"

        os.makedirs(self.processed_dir, exist_ok=True)

        # === Try to resume from checkpoint or load final output ===
        series = []
        series_with_nan = []
        start_idx = 0

        if (not rerun) and os.path.exists(ckpt_path):
            print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
            ckpt = np.load(ckpt_path, allow_pickle=True)
            # series = ckpt["series"].tolist()
            series_with_nan = ckpt["series_with_nan"].tolist()
            start_idx = int(ckpt["last_index"])
            print(f"[INFO] Resumed at iteration {start_idx}")
        elif (not rerun) and os.path.exists(path_clean) and os.path.exists(path_nan):
            print(f"[INFO] Found final output files; loading cached results.")
            return np.load(path_clean), np.load(path_nan)

        # === Load raw data ===
        print(f"[INFO] Generating new battery series (window={window_size}, tolerance={tolerance})...")
        df = pd.read_csv(self.data_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        flat_values = df.values.flatten().tolist()

        # ---- Pass 1: classify every chunk ----
        eligible_for_generation = []
        for i in range(len(flat_values) - window_size):
            chunk = flat_values[i : i + window_size]
            nan_count = np.count_nonzero(np.isnan(chunk))
            if nan_count == 0:
                chunk.insert(0, i)
                series.append(chunk)
            elif (nan_count <= tolerance) and (not np.isnan(chunk[0])) and (not np.isnan(chunk[-1])):
                eligible_for_generation.append((i, chunk))

        total_eligible = len(eligible_for_generation)
        print(f"[INFO] Clean sequences: {len(series)} | Artificial candidates: {total_eligible}")

        # ---- Pass 2: generate artificial series with tqdm from start_idx ----
        for idx, (i, chunk) in enumerate(
                tqdm(eligible_for_generation[start_idx:], desc="Generating artificial series",
                    total=total_eligible - start_idx)
            ):
            artificial_list = self.generate_artificial_battery_data(chunk, n_samples=n_samples)
            for artificial in artificial_list:
                artificial = np.insert(artificial, 0, i)
                series_with_nan.append(artificial)

            # === Checkpoint save every iteration (or choose some interval) ===
            if (idx + 1) % 1 == 0 or (idx + start_idx + 1) == total_eligible:
                try:
                    # Save to temporary path first
                    np.savez_compressed(
                        tmp_path,
                        series=np.array(series, dtype=object),
                        series_with_nan=np.array(series_with_nan, dtype=object),
                        last_index=start_idx + idx + 1
                    )
                    # Atomic replace
                    os.replace(tmp_path, ckpt_path)
                    print(f"[CHECKPOINT] Saved at iteration {start_idx + idx + 1}/{total_eligible}")
                except Exception as e:
                    print(f"[ERROR] Unable to save checkpoint at iter {idx + start_idx + 1}: {e}")

        # ---- Final conversion & save ----
        series = np.array(series, dtype=int)
        series_with_nan = np.array(series_with_nan, dtype=int)

        np.save(path_clean, series)
        np.save(path_nan, series_with_nan)

        # ---- Cleanup checkpoint since finished successfully ----
        if os.path.exists(ckpt_path):
            try:
                os.remove(ckpt_path)
                print("[INFO] Deleted checkpoint file after full completion.")
            except Exception as e:
                print(f"[WARN] Could not delete checkpoint file: {e}")

        print(f"[INFO] Saved final results → {path_clean}, {path_nan}")
        return series, series_with_nan

    # -------------------------------------------------------------------------
    # 2. Artificial data generation
    # -------------------------------------------------------------------------
    def generate_artificial_battery_data(self, seq, n_samples=25):
        """
        Generate artificial battery data using the time series generator module.
        """
        generator = tsg.Generator(
            window_size=len(seq),
            seed=seq,
            n_sample=n_samples
        )
        return generator.generate()

    # -------------------------------------------------------------------------
    # 3. Generate battery scheduling arrays
    # -------------------------------------------------------------------------
    def generate_battery_schedule(self, n_station=38 * 2, SOC_thr=0.9, window_size=48, tolerance=6):
        """
        Schedule battery usage based on generated series.
        """
        path_clean = os.path.join(self.processed_dir, f"battery_series_window{window_size}.npy")
        path_nan = os.path.join(self.processed_dir, f"battery_series_with_{tolerance}nan_window{window_size}.npy")

        if os.path.isfile(path_clean):
            battery_series = np.load(path_clean)
            series_with_nan = np.load(path_nan)
        else:
            battery_series, series_with_nan = self.generate_battery_series(
                window_size=window_size, tolerance=int(window_size / 3)
            )

        battery_series_all = np.concatenate((battery_series, series_with_nan), axis=0)
        tnum = battery_series_all.shape[1] - 1  # subtract index column

        a_vt_list, SOC_a_v_list, SOC_d_v_list, t_a_v_list, t_d_v_list = [], [], [], [], []

        for series in battery_series_all:
            a_vt, SOC_a_v, SOC_d_v, t_a_v, t_d_v = schedule_batteries(
                series, n_station, tnum, SOC_thr=SOC_thr
            )
            a_vt_list.append(np.array(a_vt, dtype=float))
            SOC_a_v_list.append(np.round(np.array(SOC_a_v, dtype=float), 2))
            SOC_d_v_list.append(np.round(np.array(SOC_d_v, dtype=float), 2))
            t_a_v_list.append(np.array(t_a_v, dtype=float))
            t_d_v_list.append(np.array(t_d_v, dtype=float))

        max_batt = max(arr.shape[0] for arr in a_vt_list)

        a_vt_list_arr = np.array([
            np.pad(arr, ((0, max_batt - arr.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
            for arr in a_vt_list
        ])
        SOC_a_v_list_arr = np.array([
            np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan)
            for arr in SOC_a_v_list
        ])
        SOC_d_v_list_arr = np.array([
            np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan)
            for arr in SOC_d_v_list
        ])
        t_a_v_list_arr = np.array([
            np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan)
            for arr in t_a_v_list
        ])
        t_d_v_list_arr = np.array([
            np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan)
            for arr in t_d_v_list
        ])

        details = np.array([SOC_a_v_list_arr, SOC_d_v_list_arr, t_a_v_list_arr, t_d_v_list_arr])

        np.save(os.path.join(self.processed_dir, f"battery_schedule_{tolerance}nan_window{window_size}.npy"), a_vt_list_arr)
        np.save(os.path.join(self.processed_dir, f"battery_details_{tolerance}nan_window{window_size}.npy"), details)

        print(f"[INFO] Saved schedule and details for window={window_size}")
        return a_vt_list_arr, details

    def generate_battery_schedule_from_custom_series(
    self,
    custom_series,
    n_station=38 * 2,
    SOC_thr=0.9,
    window_size=48,
    tolerance=6,
    save_prefix="custom",
    ):
        """
        Schedule battery usage directly from a custom input series.

        Parameters
        ----------
        custom_series : np.ndarray
            Custom battery demand time series, shape (n_series, window_size).
            Each row represents one scenario or run.
        n_station : int, default=76
            Number of battery stations or slots for scheduling.
        SOC_thr : float, default=0.9
            SOC threshold at which a battery is considered full/ready to depart.
        window_size : int, default=48
            Number of time steps in the input window.
        tolerance : int, default=6
            Tolerance parameter for naming consistency with existing outputs.
        save_prefix : str, default="custom"
            Prefix for saving output .npy files to avoid overwriting original datasets.

        Returns
        -------
        a_vt_list_arr : np.ndarray
            Array of availability schedules (n_series, n_veh, window_size).
        details : np.ndarray
            Array of details: [SOC_a_v, SOC_d_v, t_a_v, t_d_v].
        """
        assert isinstance(custom_series, np.ndarray), "custom_series must be a NumPy array."
        assert custom_series.ndim == 2, f"Expected 2D array (n_series, window_size), got {custom_series.shape}"
        assert custom_series.shape[1] == window_size+1, f"Window size mismatch ({custom_series.shape[1]} vs {window_size})"

        print(f"[INFO] Running battery scheduling for custom series: {custom_series.shape}")
        
        path_clean = os.path.join(self.processed_dir, f"{save_prefix}_battery_series_window{window_size}.npy")
        np.save(path_clean, custom_series)

        # === Initialize containers ===
        a_vt_list, SOC_a_v_list, SOC_d_v_list, t_a_v_list, t_d_v_list = [], [], [], [], []
        tnum = window_size

        for idx, series in enumerate(custom_series):
            a_vt, SOC_a_v, SOC_d_v, t_a_v, t_d_v = schedule_batteries(
                series, n_station, tnum, SOC_thr=SOC_thr
            )

            a_vt_list.append(np.array(a_vt, dtype=float))
            SOC_a_v_list.append(np.round(np.array(SOC_a_v, dtype=float), 2))
            SOC_d_v_list.append(np.round(np.array(SOC_d_v, dtype=float), 2))
            t_a_v_list.append(np.array(t_a_v, dtype=float))
            t_d_v_list.append(np.array(t_d_v, dtype=float))

            print(f"  → Series {idx+1}/{len(custom_series)} processed "
                f"(batteries: {len(SOC_a_v)} | avg SOC_a: {np.nanmean(SOC_a_v):.2f})")

        # === Padding ===
        max_batt = max(arr.shape[0] for arr in a_vt_list)
        print(f"[INFO] Max vehicle count across series: {max_batt}")

        a_vt_list_arr = np.array([
            np.pad(arr, ((0, max_batt - arr.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
            for arr in a_vt_list
        ])
        SOC_a_v_list_arr = np.array([
            np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan)
            for arr in SOC_a_v_list
        ])
        SOC_d_v_list_arr = np.array([
            np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan)
            for arr in SOC_d_v_list
        ])
        t_a_v_list_arr = np.array([
            np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan)
            for arr in t_a_v_list
        ])
        t_d_v_list_arr = np.array([
            np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan)
            for arr in t_d_v_list
        ])

        # === Combine details ===
        details = np.array([SOC_a_v_list_arr, SOC_d_v_list_arr, t_a_v_list_arr, t_d_v_list_arr])

        # === Save outputs ===
        schedule_path = os.path.join(
            self.processed_dir, f"{save_prefix}_battery_schedule_window{window_size}.npy"
        )
        details_path = os.path.join(
            self.processed_dir, f"{save_prefix}_battery_details_window{window_size}.npy"
        )
        np.save(schedule_path, a_vt_list_arr)
        np.save(details_path, details)

        print(f"[INFO] Saved custom schedule → {schedule_path}")
        print(f"[INFO] Saved custom details  → {details_path}")
        print(f"[INFO] Schedule shape: {a_vt_list_arr.shape}, Details shape: {details.shape}")

        return a_vt_list_arr, details
