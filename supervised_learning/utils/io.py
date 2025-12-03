import os
import shutil
import pandas as pd

def ensure_dir(directory: str, clean: bool = False):
    """
    Creates a directory if it doesn't exist.
    
    Args:
        directory (str): Path to the directory.
        clean (bool): If True, deletes existing files in the directory 
                      (useful for checkpoint/log folders).
    """
    if os.path.exists(directory):
        if clean:
            # Remove all files in the directory to start fresh
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(directory, exist_ok=True)

def read_clean_csv(path: str) -> pd.DataFrame:
    """
    Reads a CSV, cleans string artifacts (quotes/spaces), 
    converts to numeric, and fills missing values.
    
    Args:
        path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for processing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, index_col=0)
    
    # 1. Clean string artifacts (e.g. " 100 " -> 100)
    # We apply this to all columns to be safe
    df = df.apply(lambda c: c.astype(str).str.replace('"', '').str.strip())
    
    # 2. Convert to numeric, turning errors (like 'nan' strings) into actual NaNs
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # 3. Fill gaps (Forward fill then Backward fill)
    df = df.ffill().bfill()
    
    return df