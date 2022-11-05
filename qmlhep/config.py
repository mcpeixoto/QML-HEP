import os
from os.path import join

######################
# User Configuration #
######################

## IBM Configuration
# from qiskit import IBMQ
# token = "XXX"
# IBMQ.save_account(token)

## Define BSM Signal
# Options:
# - wohg_hq1000.h5
# - fcnc.h5
# - (...)
# More on processed_data_path or
# on https://zenodo.org/record/5126747#.YyH60tLMJcA

signal_used = "fcnc.h5"

## Maximum number of features to be used
# This is defined so that can be used in the
# Sequential Backwards Selection algorithm

max_n_features = 5

# The HP space to be tested
hp_space = {
    "feature_method": ["PCA", "SBS"],
    "n_datapoints": [5000, 1000, 500, 100],
    "n_features": [1, 2, 3, 4, 5],
    "n_layers": [1, 2, 3, 4, 5],
    "max_epochs": 500,
    "learning_rate": 0.03,
}

# CPU cores used for grid search
N_PROCESSES = os.cpu_count()

# IBM systems to be tested
ibm_systems = [
    "ibmq_lima",
    "ibmq_belem",
    "ibmq_quito",
    "ibmq_manila",
    "ibm_nairobi",
    "ibm_oslo",
]


# Decide if you want to run GPU or CPU
# on the supported models
use_gpu = False # By default is set to False to be more hardware agnostic

#######################
#        PATHS        #
#######################

paths = []
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data
data_path = join(project_root, "data")
processed_data_path = join(data_path, "processed")
raw_data_path = join(data_path, "raw")

paths += [data_path, processed_data_path, raw_data_path]

# Models
results_path = join(project_root, "results")

paths += [results_path]

# Figures
figures_path = join(project_root, "figures")

paths += [figures_path]

# Analisys results
analisys_results_path = join(results_path, "analisys_results")

paths += [analisys_results_path]

# QC Results
qc_results_path = join(results_path, "qc_results")

paths += [qc_results_path]

# Others
others_path = join(data_path, "others")

paths += [others_path]

## Create paths if they don't exist
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)
