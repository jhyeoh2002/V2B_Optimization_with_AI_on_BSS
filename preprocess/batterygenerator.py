import os
from turtle import shape
import numpy as np
import pandas as pd
from tqdm import tqdm
from preprocess.battscheduler import schedule_batteries
import config as cfg
from typing import List

import matplotlib.pyplot as plt

from time_series_generator.time_series_generator.core import Generator as BlockGenerator
from time_series_generator.time_series_generator.preprocessing import DataPrepare

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
                 tolerance: int = None):
        self.tolerance = tolerance
        self.train_path = f'./data/{cfg.BATTERYDEMAND_PATH}/tol{self.tolerance}/resample_train.csv'
        self.full_path = f'./data/{cfg.BATTERYDEMAND_PATH}/tol{self.tolerance}/resample_full.csv'
        self.battery_dir = f"./data/{cfg.BATTERYDEMAND_PATH}/tol{self.tolerance}/"
        self.resolution = cfg.RESOLUTION
        if not os.path.exists(self.full_path) or not os.path.exists(self.train_path):
            os.makedirs(os.path.dirname(self.battery_dir), exist_ok=True)
            self._preprocess_battery_data()
        else:
            print(f"\t\t[INFO] Found existing battery data at '{self.full_path}' and '{self.train_path}'")            
    
    def _load_series_from_range(self, path_template: str, index_range: range) -> pd.DataFrame:
        tdf = pd.DataFrame()
        for i in index_range:
            try:
                path = path_template.format(i)
                df = pd.read_csv(path, index_col=0)
                df.index = pd.to_datetime(df.index).floor('min')
                df = df[~df.index.duplicated()]
                tdf = pd.concat([tdf, df])
            except Exception:
                # print(f"Failed to load data from {path}")
                continue
            
        tdf = tdf[~tdf.index.duplicated()]
        tdf.sort_index(inplace=True)
        return tdf
    
    def _preprocess_battery_data(self,
        a_path_template: str = cfg.A_PATH_TEMPLATE,
        b_path_template: str = cfg.B_PATH_TEMPLATE,
        a_range: range = range(40),
        b_range: range = range(37),
        remove_dates: List[str] = cfg.TEST_DATE1 + cfg.TEST_DATE2     # <–– new parameter
        )  -> pd.Series:
        
        # 1. Load data
        a_tdf = self._load_series_from_range(a_path_template, a_range)
        b_tdf = self._load_series_from_range(b_path_template, b_range)
        tdf = pd.concat([a_tdf, b_tdf], axis=1).dropna().sum(axis=1)

        # 2. Ensure minute-level continuity
        full_index = pd.date_range(start=tdf.index.min(), end=tdf.index.max(), freq='1min')
        tdf_filled = tdf.reindex(full_index).resample(self.resolution).mean()
        tdf_filled.name = 'raw_data'

        tdf_filled.to_csv(self.full_path, index_label='timestamp')

        remove_dates = pd.to_datetime(remove_dates)
        mask = tdf_filled.index.normalize().isin(remove_dates)
        tdf_filled.loc[mask] = np.nan
            
        tdf_filled.to_csv(self.train_path, index_label='timestamp')

    # -------------------------------------------------------------------------
    # 1. Generate or load series
    # -------------------------------------------------------------------------
    def generate_battery_series(
        self,
        window_size: int = cfg.WINDOW_SIZE,
        n_samples: int = cfg.NSAMPLE,
        rerun: bool = False,
    ):
        """
        Generate battery demand series for three cases:
        Case 1: real-only clean windows
        Case 2: case1 + n_samples synthetic per eligible chunk
        Case 3: case2 + n_samples extra synthetic per eligible chunk

        Saves:
        caseX/.../battery_demand.npy
        caseX/.../battery_demand_ckpt.npz   (for crash recovery)
        """

        base_dir = self.battery_dir
        case_dirs = {
            1: os.path.join(base_dir, "case1_real_only"),
            2: os.path.join(base_dir, "case2_nan_filled"),
            3: os.path.join(base_dir, "case3_extended_generated"),
        }
        
        for d in case_dirs.values():
            os.makedirs(d, exist_ok=True)

        # === Load raw data once ===
        full_df = pd.read_csv(self.full_path, index_col=0)
        full_df.index = pd.to_datetime(full_df.index)
        full_values = full_df.values.flatten().tolist()
        
        df = pd.read_csv(self.train_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        flat_values = df.values.flatten().tolist()

        # === Pass 1: classify windows ===
        clean_series = []
        eligible_chunks = []
        
        for i in range(len(flat_values) - window_size):
            chunk = flat_values[i:i+window_size]
            nan_count = np.count_nonzero(np.isnan(chunk))
            if nan_count == 0:
                clean_series.append([i] + chunk)
            elif (nan_count <= self.tolerance) and (not np.isnan(chunk[0])) and (not np.isnan(chunk[-1])):
                eligible_chunks.append([i] + chunk)
        
        np.random.seed(cfg.RANDOM_STATE)

        generation_seed = np.random.normal(loc=28, scale=10, size=(cfg.N_EXTENDED,window_size))
        
        print(f"\n\t\t[INFO] Number of series for clean windows = {np.array(clean_series).shape}")
        print(f"\t\t[INFO] Number of eligible series for Case 2 = {np.array(eligible_chunks).shape}")
        # print(f"\t\t[INFO] Number of seeds ready for Case 3  = {np.array(generation_seed).shape}\n")

        # Helper for saving
        def _save_case(case_id, series_list):
            arr = np.array(series_list, dtype=float)
            dst = os.path.join(case_dirs[case_id], "battery_demand.npy")
            np.save(dst, arr)
            print(f"\t\t[INFO] Saved case {case_id}: '{dst}', shape={arr.shape}")
            return arr

        # -------------------------------------------------------------------------
        # Case 1 — Real only
        # -------------------------------------------------------------------------
        case1_file = os.path.join(case_dirs[1], "battery_demand.npy")
        if (not rerun) and os.path.exists(case1_file):
            case1_series = np.load(case1_file)
            print(f"\t\t[INFO] Case 1 exists — loaded battery demand from '{case1_file}' with shape {case1_series.shape}.")
        else:
            case1_series = _save_case(1, clean_series)

        # -------------------------------------------------------------------------
        # Case 2 — clean + generated (with checkpoint)
        # -------------------------------------------------------------------------
        case2_dir = case_dirs[2]
        case2_file = os.path.join(case2_dir, "battery_demand.npy")
        ckpt2_path = os.path.join(case2_dir, "battery_demand_ckpt.npz")
        tmp2_path  = os.path.join(case2_dir, "battery_demand_ckpt.tmp.npz")

        if (not rerun) and os.path.exists(case2_file):
            case2_series = np.load(case2_file)
            print(f"\t\t[INFO] Case 2 exists — loaded battery demand from '{case2_file}' with shape {case2_series.shape}.")

        else:
            print(f"\t\t[INFO] Building Case 2 (fill NaN)")
            series_case2 = case1_series.tolist()

            # ---- Checkpoint restore for case 2 ----
            start_idx = 0
            if (not rerun) and os.path.exists(ckpt2_path):
                print(f"\t\t[INFO] Resuming Case 2 from checkpoint: '{ckpt2_path}'")
                ckpt = np.load(ckpt2_path, allow_pickle=True)
                series_case2 = ckpt["series"].tolist()
                start_idx    = int(ckpt["last_index"])
                print(f"\t\t[INFO] Case 2 resume index = {start_idx}, current series shape = {np.array(series_case2).shape}")
                
            # ---- Generation loop ----
            for idx in tqdm(range(start_idx, len(eligible_chunks)),
                            desc="\t\t[INFO] Case 2 generating"):
                # i, chunk = eligible_chunks[idx][0], eligible_chunks[idx][1:]
                i, chunk = eligible_chunks[idx][0], [35.0, np.nan, 35.0, np.nan, 35.0, 24.6, 27.0, 28.0, 23.0, 23.0, 23.666666666666668, 19.0, 27.0, 27.0, 37.0, 30.0, 27.0, 23.0, 25.666666666666668, 30.0, np.nan, 33.0, 34.0, np.nan, np.nan, np.nan, 22.5, 17.5, np.nan, 19.0, 25.0, np.nan, 25.0, 25.0, 25.5, 27.0]

                sample = self._generate_artificial_battery_data(chunk, n_samples)[0]
                
                print(sample)
                
                series_case2.append([i] + np.nanmean(sample, axis=0).tolist())
                    
                self.plot_series_generator(sample, chunk, i, case_id=2)

                # ---- Save checkpoint ----
                np.savez_compressed(
                    tmp2_path,
                    series=np.array(series_case2, dtype=object),
                    last_index=idx + 1,
                )
                os.replace(tmp2_path, ckpt2_path)

            # ---- Completed → save final ----
            case2_series = _save_case(2, series_case2)
            if os.path.exists(ckpt2_path):
                os.remove(ckpt2_path)

        # # -------------------------------------------------------------------------
        # # Case 3 — extended (with checkpoint, using random seed)
        # # -------------------------------------------------------------------------
        # case3_dir = case_dirs[3]
        # case3_file = os.path.join(case3_dir, "battery_demand.npy")
        # ckpt3_path = os.path.join(case3_dir, "battery_demand_ckpt.npz")
        # tmp3_path  = os.path.join(case3_dir, "battery_demand_ckpt.tmp.npz")

        # if (not rerun) and os.path.exists(case3_file):
        #     case3_series = np.load(case3_file)
        #     print(f"\t\t[INFO] Case 3 exists — loaded battery demand from '{case3_file}' with shape {case3_series.shape}.")

        # else:
        #     print(f"\t\t[INFO] Building Case 3 (extended random-seed generation)")

        #     # Start from Case 2
        #     series_case3 = case2_series.tolist()

        #     # ---- Create random seeds for Case 3 ----
        #     generation_seed = np.random.normal(
        #         loc=28, scale=10,
        #         size=(cfg.N_EXTENDED, window_size)
        #     )

        #     print(f"\t\t[INFO] Case 3 will generate {generation_seed.shape[0]} new series from random seeds")

        #     # ---- Checkpoint restore for case 3 ----
        #     start_idx = 0
        #     if (not rerun) and os.path.exists(ckpt3_path):
        #         print(f"\t\t[INFO] Resuming Case 3 from checkpoint: '{ckpt3_path}'")
        #         ckpt = np.load(ckpt3_path, allow_pickle=True)
        #         series_case3 = ckpt["series"].tolist()
        #         start_idx    = int(ckpt["last_index"])
        #         print(f"\t\t[INFO] Case 3 resume index = {start_idx}, current series shape = {np.array(series_case3).shape}")


        #     # ---- Extended generation using random seeds ----
        #     for idx in tqdm(range(start_idx, len(generation_seed)),
        #                     desc="\t\t[INFO] Case 3 generating from random seeds"):

        #         seed = generation_seed[idx]

        #         sample = self.generate_artificial_battery_data(seed, n_samples)
                
        #         self.plot_series_generator(sample, seed, idx, case_id=3)

        #         # Append mean sample (consistent with Case 2)
        #         series_case3.append([-1] + np.nanmean(sample, axis=0).tolist())

        #         # Diagnostic plot
        #         self.plot_series_generator(sample, seed, idx)

        #         # ---- Save checkpoint ----
        #         np.savez_compressed(
        #             tmp3_path,
        #             series=np.array(series_case3, dtype=object),
        #             last_index=idx + 1,
        #         )
        #         os.replace(tmp3_path, ckpt3_path)

        #     # ---- Final save ----
        #     case3_series = _save_case(3, series_case3)
        #     if os.path.exists(ckpt3_path):
        #         os.remove(ckpt3_path)
                
        # -------------------------------------------------------------------------
        # Case 0 — Test Case
        # -------------------------------------------------------------------------
        
        full_df = pd.read_csv(self.full_path, index_col=0)
        full_df.index = pd.to_datetime(full_df.index)
        
        # 1. Define your test cases in a list to iterate easily
        test_cases = [cfg.TEST_DATE1, cfg.TEST_DATE2]
        
        # Initialize list if not already done
        test_series = [] 

        for i, (day1, day2) in enumerate(test_cases):
            
            # 2. Check if both days exist in the dataframe index
            # Using partial string indexing (e.g., .loc['2024-03-15'])
            try:
                # Extract data for the specific days
                df_d1 = full_df.loc[day1]
                df_d2 = full_df.loc[day2]
                
                # 3. Ensure both days are valid and not empty
                if not df_d1.empty and not df_d2.empty:
                    
                    # Concatenate the two days (vertical stack)
                    combined_df = pd.concat([df_d1, df_d2])
                    
                    # 4. Check length (Optional: Strict check for 48 points)
                    if len(combined_df) == 48:
                        # 2. Find the integer index (row number) of the FIRST timestamp
                        # Get the actual Timestamp object of the first row of day 1
                        first_timestamp = df_d1.index[0]
                        
                        # Find where that timestamp sits in the main dataframe (0 to len-1)
                        start_row_idx = full_df.index.get_loc(first_timestamp)
                        
                        flat_values = combined_df.values.flatten().astype(int).tolist()
                        
                        # 5. Append to your series
                        # Note: formatting as [ID, data...] to match your Case 2 structure
                        # Using 'i' as the ID (0, 1, etc.)
                        test_series.append([start_row_idx] + flat_values)
                        
                        print(f"\t\t[INFO] Case 0: Successfully added test pair {day1} & {day2}")
                        print(f"\t\tData: {test_series[-1]}")
                    else:
                        print(f"[WARNING] Date pair {day1}-{day2} found but length is {len(combined_df)}, expected 48.")
                        
            except KeyError:
                # This catches if one of the dates is completely missing from the index
                print(f"[WARNING] Skipping {day1} or {day2}: Date not found in index.")        
        
        return {1: case1_series, 2: case2_series}
    # -------------------------------------------------------------------------
    # 2. Artificial data generation
    # -------------------------------------------------------------------------
    def _generate_artificial_battery_data(self, seq, n_samples=25):
        """
        Generate artificial battery data using the time series generator module.
        """
        # generator = tsg.Generator(
        #     window_size=len(seq),
        #     seed=seq,
        #     n_sample=n_samples,
        #     tolerance=self.tolerance
        # )
        
        dataprep = DataPrepare(
            window_size=cfg.WINDOW_SIZE,
            resolution=cfg.RESOLUTION,
            tolerance=cfg.TOLERANCE_DEFAULT
        )

        # 新的呼叫方式
        generator = BlockGenerator(
            seed=seq,          
            datapreparer=dataprep,     
            window_size=cfg.WINDOW_SIZE,
            max_shift=cfg.MAX_SHIFT,
            top_k=cfg.TOP_K,
            random_state=cfg.RANDOM_STATE,
        )
        
        return generator.generate()

    # -------------------------------------------------------------------------
    # 3. Generate battery scheduling arrays
    # -------------------------------------------------------------------------
    def generate_battery_schedule(self, n_station=38 * 2, SOC_thr=0.9, window_size=48):
        """
        Schedule battery usage based on generated series.
        """
        
        base_dir = self.battery_dir
        case_dirs = {
            1: os.path.join(base_dir, "case1_real_only"),
            2: os.path.join(base_dir, "case2_nan_filled"),
            3: os.path.join(base_dir, "case3_extended_generated"),
        }
        
        for key, dir in case_dirs.items():
                
            demand_path = os.path.join(dir, f"battery_demand.npy")
            availability_path = os.path.join(dir, f"battery_availability.npy")
            details_path = os.path.join(dir, f"battery_details.npy")
            
            if os.path.exists(availability_path) and os.path.exists(details_path):
                availability = np.load(availability_path)
                details = np.load(details_path)
                print(f"\t\t[INFO] Case {key} exists — loaded from '{availability_path}' and '{details_path}' shape {availability.shape} and {details.shape}")
                continue

            try:
                battery_demand = np.load(demand_path)
            except Exception as e:
                print(f"[ERROR] Could not load battery demand from '{demand_path}': {e}")
                continue
    
            tnum = cfg.WINDOW_SIZE  # subtract index column

            a_vt_list, SOC_a_v_list, SOC_d_v_list, t_a_v_list, t_d_v_list = [], [], [], [], []

            for series in battery_demand:
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

            np.save(availability_path, a_vt_list_arr)
            np.save(details_path, details)

            print(f"\t\t[INFO] Saved availability and details for Case {key} at '{dir}'")
        
        return 

        
    def plot_series_generator(self, sample, seed, idx, case_id=2):

        mean_series = np.nanmean(sample, axis=0)
        std_series = np.nanstd(sample, axis=0)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        # --- Grey ±1 STD band (below mean and seed)
        ax.fill_between(
            np.arange(sample.shape[1]),
            mean_series - 1.96 * std_series,
            mean_series + 1.96 * std_series,
            alpha=0.3,
            color="grey",
            label="95% CI",
            linewidth=0,
            zorder=1
        )

        # --- Sample mean (blue line, middle)
        plt.plot(
            mean_series,
            label="Sample Mean",
            linewidth=1.5,
            color="blue",
            linestyle='--',
            zorder=5
        )

        # --- Seed as red X markers only (top-most)
        plt.scatter(
            np.arange(len(seed)),
            seed,
            marker="x",
            color="red",
            s=60,
            linewidths=1.5,
            label="Generation Seed",
            zorder=10
        )

        plt.xlabel("Time Step")
        plt.ylabel("Number of Fully Charged Batteries")
        plt.xticks(np.arange(0, cfg.WINDOW_SIZE + 1, step=3))
        plt.xlim(0, cfg.WINDOW_SIZE)
        plt.legend()
        plt.tight_layout()

        # Save BEFORE showing
        if case_id == 2:
            plt.savefig(f'./figures/series_generationV2_case2/{idx}.png', dpi=300)
        elif case_id == 3:
            plt.savefig(f'./figures/series_generationV2_case3/{idx}.png', dpi=300)
            
        plt.close()