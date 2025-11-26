import argparse  # <--- 1. Import argparse
from preprocess.preprocessor import DataPreprocessor
from MILP.EnergyAllocator import EnergyAllocator
import config as cfg

def main():
    print("\n", "="*50,f"PART 1: Data preprocess for {cfg.START_DATE} to {cfg.END_DATE}...", "="*50, sep="\n")
    
    ## Preprocess and generate all necessary data
    preprocessor = DataPreprocessor(cfg.START_DATE, cfg.END_DATE, args.tolerance)
    data = preprocessor.preprocess()
    
    print("\n", "="*50,f"PART 2: MILP Optimization...", "="*50, sep="\n")

    # Define which cases you want to run
    cases_to_run = [1, 2, 3] 
    
    # Use the tolerance from the command line arguments
    target_tolerance = args.tolerance 

    for case_num in cases_to_run:
        print(f"\n\t2.{case_num} Optimizing Case {case_num}...")

        # 1. Initialize Allocator for this specific case
        allocator = EnergyAllocator(case_id=case_num, tolerance=target_tolerance)
        
        # 2. Run Optimization
        allocator.run_iterations(mode="optimization", rerun=False)
        
        # 3. (Optional) Run Immediate Charging Baseline
        allocator.run_iterations(mode="immediate_charging", rerun=False)

        print(f"\t\t[INFO] Case {case_num} completed.")

if __name__ == "__main__":
    
    # --- ARGUMENT PARSING START ---
    parser = argparse.ArgumentParser(description="Run Energy Optimization and Preprocessing.")
    
    # Add the tolerance argument
    # 'type=int' ensures the input is treated as a number
    # 'default=cfg.TOLERANCE' uses your config file if the user doesn't type the flag
    parser.add_argument(
        "--tolerance", "-t", 
        type=int, 
        default=cfg.TOLERANCE_DEFAULT, 
        help=f"Target tolerance folder/value (Default: {cfg.TOLERANCE_DEFAULT})"
    )

    args = parser.parse_args()
    # --- ARGUMENT PARSING END ---
    
    main()