"""
Author: Miguel Caçador Peixoto
Description: 
    Script containing the base class for QML operatations.
"""

# Imports
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from os.path import basename, join
from datetime import datetime
from sklearn.metrics import roc_auc_score
import pickle
from functools import partial
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.embeddings import AngleEmbedding

from qmlhep.data_handling.dataset import ParticlePhysics
from qmlhep.config import results_path
from qmlhep.utils.helper import get_features

# Base class
class BaseTrainer:
    def __init__(
        self,
        study_name: str,
        name: str,
        feature_method: str,
        n_datapoints: int,
        n_features: int,
        n_layers: int,
        max_epochs: int,
        learning_rate: float,
        random_seed: int = 42,
        check_val_every_n_epoch: int = 5,
        load_data: bool = True,
        debug: bool = False,
        write_enabled: bool = True,
        **kwargs,
    ):
        """
        Base class for QML training

        Mandatory Args:
            - study_name (str)
                Name of the study, this will be used to identify the study
            - name (str)
                Name of the run, this will be used to identify the individual run of the study
            - feature_method (str)
                Method used to select features from the data
            - n_datapoints (int)
                Number of datapoints to use
            - n_features (int)
                Number of features to use
            - n_layers (int)
                Number of layers in the QML model
            - max_epochs (int)
                Number of epochs to train
            - learning_rate (float)
                Only required when using Adam optimizer,
                this will be the learning rate to use

        Optional Args:
            - random_seed (int) - Optional, default: 42
                Random seed for the experiment
            - check_val_every_n_epoch (int) - Optional, default: 5
                How often to check validation set
            - load_data (bool) - Optional, default: True
                Whether to load the data or not
            - debug (bool) - Optional, default: False
                Whether to print debug messages or not
            - write_enabled (bool) - Optional, default: True
                Whether to write to tensorboard or not
        """

        #######################
        # Defining Parameters #
        #######################
        # Put all parameters in a "hyper-parameter" dictionary for later use
        self.hp = {}
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)
                # These keys can be ignored since they don't impact results
                if key not in ["load_data", "debug", "write_enabled", "kwargs"]:
                    self.hp[key] = value

        if kwargs and self.debug:
            print("[-] Variables not in use: ", kwargs)

        # Other variables
        self.best_score = None
        self.best_score_epoch = -1
        self.best_weights = None

        #######################

        if self.debug:
            print("[+] Initializing BASE trainer...")

        #############################
        # Directories, Paths & Logs #
        #############################

        # Create directories
        self.models_directory = join(results_path, f"{self.study_name}", "models")
        self.log_directory = join(results_path, f"{self.study_name}", "logs")

        for dir in [join(results_path, f"{self.study_name}"), self.models_directory, self.log_directory]:
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                # If it errors, this is because it already exists and is caused by parallelism
                except:
                    pass

        # Paths
        self.weights_location = join(self.models_directory, f"{self.name}.wb")
        self.log_location = join(self.log_directory, self.name)
        self.info_location = join(self.models_directory, f"{self.name}_info.pkl")

        # Logs
        self.writer = SummaryWriter(log_dir=self.log_location)

        #########################
        # QuantumML Proprieties #
        #########################

        # Define device
        self.dev = qml.device("default.qubit", wires=self.n_features)

        # Embedding
        self.embedding = partial(AngleEmbedding, wires=range(self.n_features), rotation="X")
        self.normalization = "AngleEmbedding"

        # Load data
        if load_data:
            self.load_datasets()

    def get_progress(self, init_params=True):
        """
        This function checks if there was previous model progress made.
        If so, it will load the model and return the epoch and score.
        """
        # Check if there is a previous model
        if os.path.exists(self.info_location):
            # Load status
            with open(self.info_location, "rb") as f:
                info = pickle.load(f)

            # Check if info['hp'] is the same as the current hp
            assert info["hp"] == self.hp, f"[-] Parameters are not the same as previous progress! \n{info['hp']} != {self.hp}"

            # Define best score
            self.best_score = info["best_score"]
            self.best_score_epoch = info["best_score_epoch"]
            self.best_weights = info["best_weights"]

            print(
                f"[+]  {self.study_name} - {self.name}: Detected and loaded previous progress! \n\
                - Status: {info['status']} \n\
                - Best Score of {self.best_score:.4f} @ {self.best_score_epoch} epoch \n\
                - Completed {info['epoch_number']+1} epochs out of {info['hp']['max_epochs']}"
            )

            return info["epoch_number"] + 1, info["current_weights"]

        # If there isen't.. Initialize!
        else:
            if self.debug:
                print(f"[+]  {self.study_name} - {self.name}: No previous progress detected. Initializing...")

            weights = None
            if init_params:
                weights = self.init_params()
            return 0, weights

    def init_params(self):
        return NotImplementedError("[-] Init_params not implemented")

    def circuit(self, weights, x):
        return NotImplementedError("[-] Circuit not implemented")

    def activation(self, x):
        raise NotImplementedError("[-] Activation not implemented")

    def classifier(self, weights, x):
        raise NotImplementedError("[-] Classifier not implemented")

    def train(self):
        # Needs to implement self.epoch_number
        return NotImplementedError("[-] Train not implemented")

    def validation_step(self, weights):
        ###################
        # Validation Step #
        ###################

        if self.debug:
            print("[-] Validation step..")

        # Load validation data
        X_val, W_val, Y_val = self.val_dataset.all_data()

        # Remove grad
        X_val = np.array(X_val, requires_grad=False)
        Y_val = np.array(Y_val, requires_grad=False)
        W_val = np.array(W_val, requires_grad=False)

        # This will be between -1 and 1, we need to convert to between 0 and 1
        y_scores = np.array([self.classifier(weights, x) for x in X_val])
        y_scores = (y_scores + 1) / 2

        # Retormalize weights
        W_val[Y_val == 1] = (W_val[Y_val == 1] / W_val[Y_val == 1].sum()) * W_val.shape[0] / 2
        W_val[Y_val == 0] = (W_val[Y_val == 0] / W_val[Y_val == 0].sum()) * W_val.shape[0] / 2

        # Calculate ROC
        auc_score = roc_auc_score(y_true=Y_val, y_score=y_scores, sample_weight=W_val)

        # Check if new best score
        assert (self.epoch_number > self.best_score_epoch), f"[-] {self.study_name} - {self.name}: Best score epoch cannot be greater than current epoch!"
        if self.best_score is None or auc_score > self.best_score:
            if self.debug:
                print(f"{self.study_name} - {self.name} | New best score: {auc_score:0.6f} at step {self.epoch_number}")
            self.best_score = auc_score
            self.best_score_epoch = self.epoch_number
            self.best_weights = weights

            # Save model weights
            if self.write_enabled:
                with open(self.weights_location, "wb") as f:
                    pickle.dump((weights), f)

        # Log
        if self.write_enabled:
            self.writer.add_scalar("Validation AUC", auc_score, self.epoch_number)
            self.writer.add_scalar("Best_Score", self.best_score, self.epoch_number)  # Just because, probably not needed

        # Write info
        if self.write_enabled:
            self.write_info(weights)

        # Print
        if self.debug:
            print(f"[Validation] Epoch: {self.epoch_number:5d} | AUC Score: {auc_score:0.4f} | Best Score: {self.best_score:0.4f}")

        self.val_callback()

    def val_callback(self):
        """
        This function is called after each validation step
        """
        pass

    def write_info(self, weights):
        # Write Info
        if self.debug:
            print(f"[Write Info] Epoch: {self.epoch_number:5d} | Best Score: {self.best_score:0.4f} | Max Epochs: {self.max_epochs}")

        if self.epoch_number == self.max_epochs - 1:
            status = "Finished"
        else:
            status = "In Progress"

        info = {
            "status": status,
            "epoch_number": self.epoch_number,
            "current_weights": weights,
            "best_score": self.best_score,
            "best_score_epoch": self.best_score_epoch,
            "best_weights": self.best_weights,
            "hp": self.hp,
        }

        with open(self.info_location, "wb") as f:
            pickle.dump(info, f)

    def load_model(self):
        # Check if model exists
        if os.path.exists(self.weights_location):
            with open(self.weights_location, "rb") as f:
                weights = pickle.load(f)
            return weights
        else:
            raise FileNotFoundError(f"[-] Model {self.study_name} - {self.name} not found")

    def load_datasets(self):
        # Datasets initialization / Load data
        if self.debug:
            print(f"[Info] Loading data... | Random Seed: {self.random_seed}")

        ########################
        #   Dataset Specific   #
        ########################

        # Data parameters
        if self.feature_method == "SBS":
            # Get data features for training
            features = get_features(self.n_features)
            # Disable PCA features
            PCA = False
            PCA_n_features = None

        elif self.feature_method == "PCA":
            # We want to perform out PCA will a fully feature dataset
            features = None
            PCA = True
            # N components = n_features
            PCA_n_features = self.n_features
        else:
            raise ValueError("[-] Feature method not supported (yet)", self.feature_method)

        # Load data
        self.train_dataset = ParticlePhysics(
            "train",
            features=features,
            standardization=self.normalization,
            n_datapoints=self.n_datapoints,
            random_seed=self.random_seed,
            PCA=PCA,
            pca_n_features=PCA_n_features,
        )

        self.val_dataset = ParticlePhysics(
            "validation",
            features=features,
            standardization=self.normalization,
            n_datapoints=self.n_datapoints,
            random_seed=self.random_seed,
            PCA=PCA,
            pca_n_features=PCA_n_features,
        )
        
        self.test_dataset = ParticlePhysics(
            "test",
            features=features,
            standardization=self.normalization,
            n_datapoints=self.n_datapoints,
            random_seed=self.random_seed,
            PCA=PCA,
            pca_n_features=PCA_n_features,
        )

        if self.debug:
            print(f"[Info] Train dataset: {len(self.train_dataset)}")
            print(f"[Info] Validation dataset: {len(self.val_dataset)}")

    def plot_circuit(self):
        dummy_weights = np.random.randn(self.n_layers, self.n_features, 3, requires_grad=True)
        dummy_features = np.random.randn(1, self.n_features)
        fig, ax = qml.draw_mpl(qml.QNode(self.circuit, self.dev), expansion_strategy="device")(dummy_weights, dummy_features)
        return fig, ax
