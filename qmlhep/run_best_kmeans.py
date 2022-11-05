"""
Author: Miguel Caçador Peixoto
Description: 
    This script executes the kmeans study 
on VQCs and shallow CML methods.
"""

# Imports
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from os.path import join
from sklearn import svm
from utils.helper import NestablePool
import os
from tqdm import tqdm

from qmlhep.qml import AdamModel, OptunaModel
from qmlhep.utils.helper import get_kmeans_data
from qmlhep.data_handling.dataset import ParticlePhysics
from qmlhep.utils.helper import get_features
from qmlhep.config import others_path, results_path, N_PROCESSES
from qmlhep.utils.plot_results import Painter

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

########################
#### KMeans Dataset ####
########################

class KMeansParticlePhysics(ParticlePhysics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self):
        assert self.category == "train", "[!] Only train data is supported by KMeansParticlePhysics!"
        assert self.n_datapoints in [100, 500, 1000, 5000], "[!] Only 500, 1000, 5000 datapoints are supported by KMeansParticlePhysics!"

        # Shuffle it with a fixed seed (which is independent of the random seed)
        self.data = get_kmeans_data(self.n_datapoints)

        self.data = self.data.sample(frac=1, random_state=0).reset_index(drop=True)

        # Assert no Nan values
        assert not np.isnan(self.data).any().any(), "[!] Nan values found in data!"

        # Add dummy code that is expected to exist by the rest of the code
        self.data["name"] = "dummy"
        self.train_data, self.validation_data, self.test_data = self.data, 1, 1

        self.n_datapoints = None  # This way we turn off random sampling

        return self.data

####################################
#### Train the different Models ####
####################################

def get_kmeans_results(hp):
    hp["name"] = f'KMEANS_{hp["regime"]}_{hp["n_datapoints"]}_{hp["random_seed"]}'

    # Get model for training
    trainer = globals()[hp["study_name"]](**hp)

    # Data parameters
    if trainer.feature_method == "SBS":
        # Get data features for training
        features = get_features(trainer.n_features)
        # Disable PCA features
        PCA = False
        PCA_n_features = None

    elif trainer.feature_method == "PCA":
        # We want to perform out PCA will a fully feature dataset
        features = None
        PCA = True
        # N components = n_features
        PCA_n_features = trainer.n_features
    else:
        raise ValueError("[-] Feature method not supported (yet)", trainer.feature_method)

    if hp["regime"] == "kmeans":
        trainer.train_dataset = KMeansParticlePhysics(
            "train",
            features=features,
            standardization=trainer.normalization,
            n_datapoints=trainer.n_datapoints,
            random_seed=trainer.random_seed,
            PCA=PCA,
            pca_n_features=PCA_n_features,
        )
        
    elif hp["regime"] == "regular":
        trainer.train_dataset = ParticlePhysics(
            "train",
            features=features,
            standardization=trainer.normalization,
            n_datapoints=trainer.n_datapoints,
            random_seed=trainer.random_seed,
            PCA=PCA,
            pca_n_features=PCA_n_features,
        )
    else:
        raise ValueError("[-] Work regime not supported:", hp["regime"])

    #################################
    #### QML Training
    #################################

    # Train QML
    trainer.train()

    # Load weights
    weights = trainer.load_model()

    #### Test data ####
    X_test, W_test, Y_test = trainer.test_dataset.all_data()

    # Retormalize weights
    W_test[Y_test == 1] = (W_test[Y_test == 1] / W_test[Y_test == 1].sum()) * W_test.shape[0] / 2
    W_test[Y_test == 0] = (W_test[Y_test == 0] / W_test[Y_test == 0].sum()) * W_test.shape[0] / 2

    #  Compute predictions
    y_test_scores = np.array([trainer.classifier(weights, x) for x in X_test])
    y_test_scores = (y_test_scores + 1) / 2

    # Compute AUC
    fpr, tpr, _ = roc_curve(Y_test, y_test_scores, sample_weight=W_test)
    qml_roc_auc = auc(fpr, tpr)

    #################################
    #### Shallow ML Training
    #################################

    trainer = globals()[hp["study_name"]](**hp, load_data=False)
    trainer.normalization = "ML"
    trainer.load_datasets()

    if hp["regime"] == "kmeans":
        trainer.train_dataset = KMeansParticlePhysics(
            "train",
            features=features,
            standardization="ML",
            n_datapoints=trainer.n_datapoints,
            random_seed=trainer.random_seed,
            PCA=PCA,
            pca_n_features=PCA_n_features,
        )
    elif hp["regime"] == "regular":
        trainer.train_dataset = ParticlePhysics(
            "train",
            features=features,
            standardization="ML",
            n_datapoints=trainer.n_datapoints,
            random_seed=trainer.random_seed,
            PCA=PCA,
            pca_n_features=PCA_n_features,
        )
    else:
        raise ValueError("[-] Work regime not supported:", hp["regime"])

    #### Train data ####
    X_train, W_train, Y_train = trainer.train_dataset.all_data()

    # Retormalize weights
    W_train[Y_train == 1] = (W_train[Y_train == 1] / W_train[Y_train == 1].sum()) * W_train.shape[0] / 2
    W_train[Y_train == 0] = (W_train[Y_train == 0] / W_train[Y_train == 0].sum()) * W_train.shape[0] / 2

    #### Test data ####
    X_test, W_test, Y_test = trainer.test_dataset.all_data()

    # Retormalize weights
    W_test[Y_test == 1] = (W_test[Y_test == 1] / W_test[Y_test == 1].sum()) * W_test.shape[0] / 2
    W_test[Y_test == 0] = (W_test[Y_test == 0] / W_test[Y_test == 0].sum()) * W_test.shape[0] / 2

    #################
    ### Train SVM ###
    #################
    clf = svm.SVC(kernel="rbf", probability=True)

    clf.fit(X_train, Y_train, sample_weight=W_train)

    # Predict
    y_test_scores = clf.predict_proba(X_test)
    y_test_scores = y_test_scores[:, 1]

    # Compute AUC
    fpr, tpr, _ = roc_curve(Y_test, y_test_scores, sample_weight=W_test)
    svm_roc_auc = auc(fpr, tpr)

    #################
    ### Train LR  ###
    #################
    clf = LogisticRegression()

    clf.fit(X_train, Y_train, sample_weight=W_train)

    # Predict
    y_test_scores = clf.predict_proba(X_test)
    y_test_scores = y_test_scores[:, 1]

    # Compute AUC
    fpr, tpr, _ = roc_curve(Y_test, y_test_scores, sample_weight=W_test)
    lr_roc_auc = auc(fpr, tpr)

    #################
    ### Concatenate results
    #################
    results = []
    for model in ["qml", "svm", "lr"]:
        _hp = hp.copy()
        _hp["regime"] = hp["regime"]
        _hp["model"] = model
        _hp["auc"] = eval(f"{model}_roc_auc")
        results.append(_hp)

    return results


if __name__ == "__main__":
    painter = Painter(use_test_data=True)

    # Load the best set of hyperparameters
    model = "OptunaModel"
    best_name, best_run, (qml_worlds, svm_worlds, log_worlds) = painter.get_best_name(model)
    hp = qml_worlds[0]

    # Delete useless keys
    for key in ["X_val", "Y_val", "W_val", "Y_val_scores", "X_test", "Y_test", "W_test", "Y_test_scores"]:
        del hp[key]

    # Create a workload
    workload = []
    seeds = [world["random_seed"] for world in qml_worlds]
    for regime in ["kmeans", "regular"]:
        for n_datapoints in [100, 500, 1000, 5000]:
            for seed in seeds:
                hp["n_datapoints"] = n_datapoints
                hp["random_seed"] = seed
                hp["regime"] = regime
                workload.append(hp.copy())

    with NestablePool(processes=N_PROCESSES) as p:
        r = list(tqdm(p.imap(get_kmeans_results, workload), total=len(workload), desc=f"[KMEANS STUDY] Waiting for processes to finish..."))

        df = pd.DataFrame([item for sublist in r for item in sublist])
        df.to_csv(join(results_path, "kmeans_results.csv"), index=False)
