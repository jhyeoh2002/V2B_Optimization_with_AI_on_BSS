from preprocess.preprocessor import DataPreprocessor
import config as cfg

def main():
    
    ## Preprocess and generate all necessary data
    preprocessor = DataPreprocessor(cfg.START_DATE, cfg.END_DATE)
    data = preprocessor.preprocess()
    

if __name__ == "__main__":
    main()