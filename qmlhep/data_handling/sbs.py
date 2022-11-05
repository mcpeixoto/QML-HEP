"""
Author: Miguel Caçador Peixoto
Description: 
    This script is contains the Sequential Backwards Selection algorithm,
which is used to select the best features for a given dataset.

For more information:
- https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
- http://rasbt.github.io/mlxtend/api_modules/mlxtend.feature_selection/SequentialFeatureSelector/
"""

# Imports
from tqdm import tqdm
import os
from os.path import join
import pickle
from xgboost import XGBClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import torch
from datetime import datetime
from qmlhep.config import max_n_features, others_path, use_gpu
from qmlhep.data_handling.dataset import ParticlePhysics

# Load data
data = ParticlePhysics(category="train", standardization="ML").all_data_Dataframe()
data.drop(columns=["name"], inplace=True)
features = data.columns[:-2]

X, y, w = data[features], data["label"], data["weights"]

# Renormalize weights
w[y == 1] = (w[y == 1] / w[y == 1].sum()) * w.shape[0] / 2
w[y == 0] = (w[y == 0] / w[y == 0].sum()) * w.shape[0] / 2

# Select tree method based on hardware available
# GPU
if use_gpu:
    tree_method = 'gpu_hist'
else:
    tree_method = 'hist'

    if torch.cuda.is_available():
        print("GPU is available but set to False in config.py. It's very likely that this will take a long time!")

# Initialize a dictionary to save the results
# keys -> number of features
# values -> list of features
book = {}

# Start the algorithm
print(f"[+] {datetime.now()} - Starting Sequential Backwards Selection with {max_n_features} features")
for n_features in tqdm(range(max_n_features, 0, -1), desc="[SBS] Running.."):
    # Select the features
    X = X[features]

    # Initialize the classifier
    clf = XGBClassifier(
        n_estimators=100,
        learning_rate=1e-5,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=-1,
        tree_method=tree_method,
    )

    # Sequential Forward Floating Selection
    sffs = SFS(
        clf,
        k_features=n_features,
        forward=False,
        scoring="roc_auc",
        cv=4,
        n_jobs=os.cpu_count(),
    )
    
    # Fit
    sffs = sffs.fit(X, y, feature_weights=w)

    # Print
    print("-" * 50)
    print(f"#Features: {n_features}")
    print(f"CV Score: {sffs.k_score_}")
    print(f"Selected features: {sffs.k_feature_names_}")

    # Add to book
    features = list(sffs.k_feature_names_)
    book[n_features] = features


# Save book
with open(join(others_path, "SBS.pkl"), "wb") as f:
    pickle.dump(book, f)

print(f"[+] {datetime.now()} - Finished Sequential Backwards Selection")
