"""
Author: Miguel Caçador Peixoto
Description: 
    This script is used to preprocess the data the raw data from zenodo
and dump the processed data into a new folder
"""

# Imports
import pandas as pd
from glob import glob
import os
from os.path import join, basename
from tqdm import tqdm
from multiprocessing import Pool

from qmlhep.config import raw_data_path, processed_data_path

# Listing all the files on the bkg and signal directory (file path)
bkg_files = glob(join(raw_data_path, "bkg_*"))
signal_files = list(set(glob(join(raw_data_path, "*.*"))) - set(bkg_files))
all_files = bkg_files + signal_files

print("Found files:\n-#Signal:", len(signal_files), "\n-#Background:", len(bkg_files), "\n> Total:", len(all_files))

## Sanity checks

# Check if they all have the same features
# If not, identify the problematic features
# and remove them from the data

book = {}
for path in all_files:
    # Load data
    data = pd.read_hdf(path)

    # Get features
    features = list(data.columns)

    file_name = basename(path)
    for feature in features:
        if feature not in book:
            book[feature] = []
        book[feature] += [file_name]

problematic_features = []
for x in book:
    if x.startswith("gen"):
        problematic_features.append(x)
    elif len(book[x]) != len(all_files):
        problematic_features.append(x)
        # print(f'\nFeature "{x}" is missing on', len(all_files) - len(book[x]), "file(s).")
        # print("-> Files that are missing the feature:\n\t", set([basename(x) for x in all_files]) - set(book[x]))
    else:
        pass

# This features shall be removed
# They are comprised by the problematic features
# plus features that were used in data generation
to_remove = list(
    set(
        [
            "gen_decay1",
            "gen_decay2",
            "gen_sample",
            "gen_filter",
            "gen_decay_filter",
            "MissingET_Eta",
            "gen_label",
            "gen_n_btags",
            "gen_sample_filter",
            "gen_split",
        ]
        + problematic_features
    )
)

# This feature is the weights
to_remove.remove("gen_xsec")


## Pre-Processing
# -> Delete irrelevant features (ex: features that were used on the simulation of the data)
# -> Apply data cuts
# - At least two final state leptons
# - At least one b-tagged jet
# - Large scalar sum of transverse momentum (p_t), H_t > 500 GeV
# -> Preprocess monte carlo weights
# -> On the BKG data, only use the subset Zbb


def pre_process(path):
    # Clean filename
    file_name = basename(path).replace("_pythia_sanitised_features", "")

    # Load data
    data = pd.read_hdf(path, index_col=0)

    # Restringe data
    if "bkg" in file_name:
        data = data[data["gen_sample"] == "Zbb"]

    # Remove useless features
    for x in to_remove:
        try:
            data.drop([x], axis=1, inplace=True)
        except:
            pass

    data = data.astype("float32")

    ## Apply Cuts
    # Statistical purposes
    shape_before = data.shape[0]
    # At least 2 leptons
    data = data[(data["Electron_Multi"] + data["Muon_Multi"]) >= 2]
    # At least 1 B-Tag
    data = data[(data["Jet1_BTag"] + data["Jet2_BTag"] + data["Jet3_BTag"] + data["Jet4_BTag"] + data["Jet5_BTag"]) >= 1]
    # H_T > 500 GeV
    data = data[data["ScalarHT_HT"] > 500]

    shape_after = data.shape[0]
    print(f'[Info] Data Reduction for "{file_name}": {int(((shape_before-shape_after)/shape_before)*100)}%')

    ## Monte Carlo Weights preprocess
    data["gen_xsec"] = data["gen_xsec"].mean() / data.shape[0]
    data.rename(columns={"gen_xsec": "weights"}, inplace=True)

    # Add a column with the file name
    data["name"] = file_name

    # Give label to data
    if "bkg" in file_name:
        data["label"] = 0
    else:
        data["label"] = 1

    # Reset index
    data.reset_index(inplace=True, drop=True)

    ## SAVING
    data.to_hdf(join(processed_data_path, file_name), key=file_name.replace(".h5", ""), mode="w")


if __name__ == "__main__":
    print("[Info] Started preprocessing..")

    # Multiprocessing
    with Pool(processes=os.cpu_count()) as p:
        r = list(tqdm(p.imap(pre_process, all_files), total=len(all_files), desc="Waiting for processes to finish..."))

    print("[Info] Finished preprocessing..")
