from preprocess.preprocessor import DataPreprocessor
from optimization.util.EnergyAllocator import EnergyAllocator
import config as cfg

def main():
    
    ## Preprocess and generate all necessary data
    preprocessor = DataPreprocessor(cfg.START_DATE, cfg.END_DATE)
    data = preprocessor.preprocess()
    
    energy_allocator = EnergyAllocator()
    results = energy_allocator.run_iterations(mode="optimization")
    results = energy_allocator.run_iterations(mode="immediate_charging")
    

if __name__ == "__main__":
    main()