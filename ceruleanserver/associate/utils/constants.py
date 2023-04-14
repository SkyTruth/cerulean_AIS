import numpy as np
import os


# for leaflet application
DATA_DIR = '/home/k3blu3/datasets/cerulean'
AIS_DIR = os.path.join(DATA_DIR, '19_ais')
SLICK_DIR = os.path.join(DATA_DIR, '19_vectors')
TRUTH_FILE = os.path.join(DATA_DIR, 'slick_truth_year1.csv')

# temporal parameters for AIS trajectory estimation
HOURS_BEFORE = 12
NUM_TIMESTEPS = 120

# buffering parameters for AIS trajectories
BUF_START = 5000
BUF_END = (BUF_START * HOURS_BEFORE) / 4
BUF_VEC = np.linspace(BUF_START, BUF_END, NUM_TIMESTEPS)

# weighting parameters for AIS trajectories
WEIGHT_START = 1.0
WEIGHT_END = 0.1
WEIGHT_VEC = np.logspace(WEIGHT_START, WEIGHT_END, NUM_TIMESTEPS) / 10.0
