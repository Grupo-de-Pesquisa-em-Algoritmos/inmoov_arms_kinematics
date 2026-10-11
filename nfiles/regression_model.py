import pandas as pd
import numpy as np
from data_split import create_spatial_split
from plot_experiments import get_eff_pos
from data_utils import read_dataset
FILENAME = "dataset_regressao_energia.csv"


X, y = read_dataset(FILENAME)
pos_exp = get_eff_pos(X, y)
splits = create_spatial_split(X, y, pos_exp)

idx_dev = splits["idx_dev"]
idx_test = splits["idx_test"]
folds = splits["folds"]
