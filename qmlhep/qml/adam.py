"""
Author: Miguel Caçador Peixoto
Description: 
    Script containing the adam-optimizer qml class implementation.
"""

# Imports
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

from qmlhep.utils.helper import square_loss
from qmlhep.qml.base import BaseTrainer


class AdamModel(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.debug:
            print("[+] Initializing AdamModel class..")

    def init_params(self):
        # Random weight initialization
        weights = 0.01 * np.random.randn(self.n_layers, self.n_features, 3, requires_grad=True)
        return weights

    def circuit(self, weights, x):
        # Embedding
        self.embedding(x)

        # For every layer
        for layer in range(self.n_layers):
            W = weights[layer]

            # Define Rotations
            for i in range(self.n_features):
                qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)

            # Entanglement
            if self.n_features != 1:
                if self.n_features > 2:
                    for i in range(self.n_features):
                        if i == self.n_features - 1:
                            qml.CNOT(wires=[i, 0])
                        else:
                            qml.CNOT(wires=[i, i + 1])
                else:
                    qml.CNOT(wires=[1, 0])

        return qml.expval(qml.PauliZ(0))

    def classifier(self, weights, x):
        return qml.QNode(self.circuit, self.dev)(weights, x)

    def train(self):
        # Initialize optimizer
        self.opt = AdamOptimizer(self.learning_rate)

        # Initilize
        self.epoch_number, weights = self.get_progress()

        # For every epoch
        for epoch in range(self.epoch_number, self.max_epochs):
            self.epoch_number = epoch

            # Update weights
            loss, weights = self.train_step(weights)

            if self.debug:
                print(f"[+] Epoch: {epoch} |  Loss: {loss:.4f}")

            # Write loss
            if self.write_enabled:
                self.writer.add_scalar("Loss", loss, self.epoch_number)

            # Validate the model
            if self.epoch_number % self.check_val_every_n_epoch == 0 or self.epoch_number == self.max_epochs - 1:
                self.validation_step(weights)

        print("[+] Training finished")

        return self.best_score

    def train_step(self, weights):
        X_train, W_train, Y_train = self.train_dataset.all_data()

        # Only require grad if necessary
        X_train = np.array(X_train, requires_grad=False)
        Y_train = np.array(Y_train, requires_grad=True)
        W_train = np.array(W_train, requires_grad=False)

        # Compute cost and update weights
        weights, loss = self.opt.step_and_cost(self.cost, weights, X=X_train, Y=Y_train, W=W_train)

        return loss, weights

    def cost(self, weights, X=None, Y=None, W=None):  # X, Y, W are keyword arguments so that they are ignored by optimizer
        # Compute predictions
        y_scores = [(self.classifier(weights, x) + 1) / 2 for x in X]

        # Retormalize data weights
        W[Y == 1] = (W[Y == 1] / W[Y == 1].sum()) * W.shape[0] / 2
        W[Y == 0] = (W[Y == 0] / W[Y == 0].sum()) * W.shape[0] / 2

        loss = square_loss(Y, y_scores)
        loss = loss * W
        loss = loss.sum()

        return loss

    def __name__(self):
        return "AdamModel"
