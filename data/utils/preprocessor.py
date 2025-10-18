import os
import sys
import pandas as pd

# Add the project root directory to the Python module search path
sys.path.append(os.path.abspath(".."))

class DataPreprocessor:
    def __init__(self):
        pass

    def generate_weather_and_radiation(self):
        df = pd.read_csv('data/rawdata/466920-2023-05-GlobalSolarRadiation-hour.csv')
        
        
        
        return df
    
    
