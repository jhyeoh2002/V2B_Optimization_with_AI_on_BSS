import sys
import os

# Add the project root directory to the Python module search path
sys.path.append(os.path.abspath(".."))

# Import core module and configuration
import time_series_generator.core as tsg
import time_series_generator.config as cfg

# Initialize the generator with configuration parameters
generator = tsg.Generator(
    window_size=cfg.WINDOW_SIZE,       # Length of each time series subsequence (default: 24)
    resolution=cfg.RESOLUTION,         # Time resolution of the data (default: '1h')
    seed=cfg.SEED,                     # Input seed sequence (default: sampled from N(mean=40, std=20, size=window_size))
    n_sample=cfg.NSAMPLE               # Number of new samples to generate (default: 500)
)

# Generate samples from the estimated posterior distribution
sample = generator.generate()


