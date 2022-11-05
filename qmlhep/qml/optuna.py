"""
Author: Miguel Caçador Peixoto
Description: 
    Script containing the optuna-optimizer qml class implementation.
"""

# Imports
import os
from os.path import join
from pennylane import numpy as np
import optuna
import pickle
import warnings

from qmlhep.qml.adam import AdamModel

class OptunaModel(AdamModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.debug:
            print("[+] Initializing OptunaModel class..")

        # Silence optuna if debug is disabled
        if not self.debug:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            warnings.filterwarnings("ignore")

        # Initialize optuna-specific paths
        self.study_path = join(self.models_directory, f"{self.name}.db")
        self.study_path_checkpoint = join(self.models_directory, f"{self.name}_checkpoint.db")

    def init_params(self, inner_trial):
        # Weight initialization with optuna
        weights = np.ones((self.n_layers, self.n_features, 3), dtype=np.float32, requires_grad=True)
        for layer in range(self.n_layers):
            for qubit in range(self.n_features):
                for i in range(3):
                    weights[layer, qubit, i] = inner_trial.suggest_uniform(f"W_{layer}_{qubit}_{i}", -np.pi, np.pi)
        return weights

    def train(self):
        # Initialize OptunaModel
        self.epoch_number, _ = self.get_progress(init_params=False)

        if self.debug:
            print(f"[+] Starting OptunaModel training for in {self.epoch_number} epochs of a maximum of {self.max_epochs} epochs")

        # Optimize
        self.optimizer.optimize(self.inner_objective, n_trials=self.max_epochs - self.epoch_number)

        return self.best_score

    def inner_objective(self, inner_trial):
        # Guess weights with OptunaModel
        weights = self.init_params(inner_trial)

        # Calculate train loss for inner study
        loss, _ = self.train_step(weights)

        # Call validation step
        # This will be used to calculate the validation AUC score,
        # report back to the main study and to save the best weights
        if self.epoch_number % self.check_val_every_n_epoch == 0 or self.epoch_number == self.max_epochs - 1:
            self.validation_step(weights)

        # Update epoch (for reporting)
        self.epoch_number += 1

        return loss

    def train_step(self, weights):
        X_train, W_train, Y_train = self.train_dataset.all_data()

        X_train = np.array(X_train, requires_grad=False)
        W_train = np.array(W_train, requires_grad=False)
        Y_train = np.array(Y_train, requires_grad=False)

        loss = self.cost(weights, X=X_train, Y=W_train, W=Y_train)

        if self.write_enabled:
            self.writer.add_scalar("loss", loss, self.epoch_number)

        if self.debug:
            print(f"[+] Epoch: {self.epoch_number} | Train Loss: {loss:.4f}")

        return loss, weights

    def get_progress(self, init_params=True):
        """
        Overwriting the original get_progress method
        Apart from loading the model, this will also load & checkpoint optuna study
        """

        # Check if there is a previous model
        if os.path.exists(self.info_location):
            # Load status
            with open(self.info_location, "rb") as f:
                info = pickle.load(f)

            # Check if info['hp'] is the same as the current hp
            assert info["hp"] == self.hp, f"[-] HP is not the same as previous progress! \n{info['hp']} != {self.hp}"

            # Define best score
            self.best_score = info["best_score"]
            self.best_score_epoch = info["best_score_epoch"]
            self.best_weights = info["best_weights"]

            # if self.debug:
            print(
                f"[+]  {self.study_name} - {self.name}: Detected and loaded previous progress! \n\
                - Status: {info['status']} \n\
                - Best Score of {self.best_score:.4f} @ {self.best_score_epoch} epoch \n\
                - Completed {info['epoch_number']+1} epochs out of {info['hp']['max_epochs']}"
            )

            ################## OPTUNA SPECIFIC CODE ##################

            # Copy checkpoint to current study database
            os.system(f"cp {self.study_path_checkpoint} {self.study_path}")

            # Initialize optimizer
            study_names = [study.study_name for study in optuna.get_all_study_summaries(storage="sqlite:///" + self.study_path)]
            assert len(study_names) == 1, f"[-] More than one study found in {self.study_path}! {study_names}"
            self.optimizer = optuna.load_study(
                study_name=study_names[0],
                storage="sqlite:///" + self.study_path,
            )

            ################# # ################## # #################

            return info["epoch_number"] + 1, info["current_weights"]

        # If there isen't.. Initialize!
        else:
            if self.debug:
                print(f"[+] Run: {self.name} - No previous progress detected. Initializing...")

            ################## OPTUNA SPECIFIC CODE ##################

            # Initialize optimizer
            self.optimizer = optuna.create_study(
                direction="minimize", study_name=self.name, storage="sqlite:///" + self.study_path, load_if_exists=False
            )

            ################# # ################## # #################

            weights = None
            if init_params:
                weights = self.init_params()
            return 0, weights

    def val_callback(self):
        # Checkpoint optuna study db
        os.system(f"cp {self.study_path} {self.study_path_checkpoint}")

    def __name__(self):
        return "OptunaModel"
