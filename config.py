# Parameters for OpenPowerDataset
N_CLASSES = 3
THRESHOLD_MODE = "manual"
MANUAL_THRESHOLDS = [10, 11.5]

# Paths for data_preparation.py
SOLAR_POWER_FILE = 'data/open_power/solar_15min.csv'
WEATHER_FILE = 'data/open_power/weather_data.csv'

# Column name for data_preparation.py
COLUMN_NAME = "utc_timestamp"

N_EPOCHS = 10
BATCH_SIZE = 64
LR = 0.0002
B1 = 0.5
B2 = 0.999
N_CPU = 8
LATENT_DIM = 100
SERIES_LENGTH = 96
N_CRITIC = 5
CLIP_VALUE = 0.01
DATASET = 'open_power'
SAMPLE_INTERVAL = 1000