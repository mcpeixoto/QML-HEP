"""
Author: Miguel Caçador Peixoto
Description: 
    This script contains the main Dataset Class, ParticlePhysics.
"""


# Imports
from torch.utils.data import Dataset
import pandas as pd
from os.path import basename, join
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import numpy as np
import gc

from qmlhep.config import signal_used, processed_data_path

# Removed SettingWithCopyWarning
pd.options.mode.chained_assignment = None


class ParticlePhysics(Dataset):
    """
    This class will load, split, transform and standardize the data given the specified parameters.
    """

    def __init__(self, category, features=None, PCA=False, pca_n_features=None, standardization="ML", n_datapoints=None, random_seed=42, debug=False):

        # Defining Parameters
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)

        # Sanity checks
        assert self.category in {"train", "validation", "test", "all"}, "[!] Invalid category!"
        assert self.standardization in {"ML", "AngleEmbedding"}, "[!] Invalid standardization! Standardization must be either 'ML' or 'AngleEmbedding'"
        if not self.PCA and self.pca_n_features is not None:
            raise ValueError("pca_n_features must be None if PCA is False!")
        if self.PCA and self.features is not None and self.pca_n_features is not None:
            assert self.pca_n_features <= len(self.features), "[!] pca_n_features must be less or equal to the number of features!"

        # Set random seed
        if self.random_seed is not None:
            if self.debug:
                print(f"[{self.category}] Random Seed: {self.random_seed}")
            np.random.seed(self.random_seed)

        # Load data
        self.load_data()

        # Filter the relevant features
        if self.features is not None:
            self.data = self.data[list(set(list(self.features) + ["weights", "name", "label"]))]

        # Perform PCA
        if self.PCA:
            self.perform_PCA()

        # Standardization of the data
        self.DataFeatures = pd.Index(
            list(set(self.data.columns) - set(["weights", "name", "label"]))
        )  # Everything except the weights, name and label

        if self.standardization == "ML":
            self.data[self.DataFeatures] = (self.data[self.DataFeatures] - self.data[self.DataFeatures].mean()) / self.data[self.DataFeatures].std()
        elif self.standardization == "AngleEmbedding":
            # Put the data between -pi and pi
            self.data[self.DataFeatures] = (((self.data[self.DataFeatures] - self.data[self.DataFeatures].min()) / (self.data[self.DataFeatures].max() - self.data[self.DataFeatures].min())) * 2 - 1) * (np.pi)
        else:
            # Error
            raise ValueError("Invalid standardization!")

        if self.debug:
            print("[INFO] Considered features: {}".format(", ".join(self.DataFeatures)))

        # Sample the n_datapoints
        # This will be done taking in consideration the % of signal and bkg
        # in order to keep the statistical significance of the data.
        if self.n_datapoints is not None:
            sig_pctg = len(self.data[self.data["label"] == 1]) / len(self.data)

            # Sample the signal and background data
            sig_data = self.data[self.data["label"] == 1].sample(n=int(self.n_datapoints * sig_pctg), random_state=self.random_seed)
            bkg_data = self.data[self.data["label"] == 0].sample(n=int(self.n_datapoints * (1 - sig_pctg)), random_state=self.random_seed)

            # Concatenate the data
            self.data = pd.concat([sig_data, bkg_data], axis=0)

        ## Split the data into diferent variables
        # (This data we want on seperate variables)
        self.weights = self.data["weights"]
        self.name = self.data["name"]
        self.label = self.data["label"]
        self.data.drop(columns=["name", "weights", "label"], inplace=True)
        self.n_samples = self.data.shape[0]

        assert self.data.isnull().values.any() == False, "[!] Data contains NaN values!"

        if self.debug:
            print("[{}] Loaded {} samples. Wanted {}".format(self.category, self.n_samples, self.n_datapoints))

        # Clean
        del self.train_data, self.validation_data, self.test_data
        gc.collect()

    def load_data(self):
        """
        Loads the data from the disk, shuffles it and splits it into train, validation and test.
        This will create self.train_data, self.validation_data and self.test_data.
        self.data will be set to the desired category
        """
        # Load from disk
        data = []
        for file in [join(processed_data_path, "bkg.h5"), join(processed_data_path, signal_used)]:
            if self.debug is True:
                print("[INFO] Loading data from {}".format(basename(file)))
            data.append(pd.read_hdf(file))
        data = pd.concat(data)

        # Shuffle it with a fixed seed (which is independent of the random seed)
        self.data = data.sample(frac=1, random_state=0).reset_index(drop=True)

        # Split into train, validation and test
        # This will take in consideration the statistical distribution of the data
        signal_data = self.data[self.data["label"] == 1]
        bkg_data = self.data[self.data["label"] == 0]

        sig_train, sig_validate, sig_test = np.split(signal_data, [int((1 / 3) * len(signal_data)), int((2 / 3) * len(signal_data))])
        bkg_train, bkg_validate, bkg_test = np.split(bkg_data, [int((1 / 3) * len(bkg_data)), int((2 / 3) * len(bkg_data))])

        # Define the diferent datasets
        self.train_data = pd.concat([sig_train, bkg_train])
        self.validation_data = pd.concat([sig_validate, bkg_validate])
        self.test_data = pd.concat([sig_test, bkg_test])

        if self.category == "train":
            self.data = self.train_data
        elif self.category == "validation":
            self.data = self.validation_data
        elif self.category == "test":
            self.data = self.test_data
        elif self.category == "all":
            self.data = pd.concat([self.train_data, self.validation_data, self.test_data])
        else:
            raise ValueError("Invalid category!")
        del sig_train, sig_validate, sig_test, bkg_train, bkg_validate, bkg_test

        if self.debug is True:
            print("[INFO] Loaded {} samples".format(len(data)))

        # Reset indexes
        self.data.reset_index(drop=True, inplace=True)
        self.train_data.reset_index(drop=True, inplace=True)
        self.validation_data.reset_index(drop=True, inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)

        return data

    def perform_PCA(self):
        # which data will be used for fitting the PCA.
        # Everything except the weights, name and label
        self.DataFeatures = pd.Index(list(set(self.data.columns) - set(["weights", "name", "label"])))

        ## Fit PCA to train data & rank components by AUC
        pca = PCA(n_components=len(self.DataFeatures))
        pca.fit(self.train_data[self.DataFeatures])

        ## Transform the desired dataset to get its principal components
        # Get ranked components by AUC from the train data
        principalComponents = pca.transform(self.train_data[self.DataFeatures])

        # Book will be a dictiorary with the AUC (values) of each component (keys)
        book = {}

        # Get values for AUC computation
        y_true = self.train_data["label"].values
        weights = self.train_data["weights"].values

        # Renormalise weights
        weights[y_true == 1] = (weights[y_true == 1] / weights[y_true == 1].sum()) * weights.shape[0] / 2
        weights[y_true == 0] = (weights[y_true == 0] / weights[y_true == 0].sum()) * weights.shape[0] / 2

        for feature_idx in range(principalComponents.shape[1]):
            book[f"Component {feature_idx}"] = roc_auc_score(y_true=y_true, y_score=principalComponents[:, feature_idx], sample_weight=weights)

        # Give me the best features
        book = pd.DataFrame.from_dict(book, orient="index")
        book.columns = ["AUC"]
        book.sort_values(by="AUC", ascending=False, inplace=True)
        book.reset_index(inplace=True)
        book.rename(columns={"index": "Feature"}, inplace=True)

        ## Replace current data by its components ##
        # Get components for the current set we want
        principalComponents = pca.transform(self.data[self.DataFeatures])

        # Create a new dataframe with PCA data
        newdf = pd.DataFrame(principalComponents, columns=[f"Component {i}" for i in range(principalComponents.shape[1])])

        # Select the best components given their AUC performance in training data
        if self.pca_n_features:
            newdf = newdf[book["Feature"][0 : self.pca_n_features]]

        # Add the other relevant features
        newdf["weights"] = self.data["weights"].values
        newdf["name"] = self.data["name"].values
        newdf["label"] = self.data["label"].values

        # Finally, replace self.data with newdf
        self.data = newdf

        # Update DataFeatures
        self.DataFeatures = pd.Index(list(set(self.data.columns) - set(["weights", "name", "label"])))

        if self.debug:
            print("[INFO] PCA performed")
            print("[INFO] New features: {}".format(self.DataFeatures))

    def get_columns(self):
        return self.data.columns

    def all_data_Dataframe(self):
        # Append data, weights, label, name to the dataframe and return it
        return pd.concat([self.data, self.weights, self.label, self.name], axis=1)

    def all_data(self):
        # Note: Name column is irrelevant, so is not returned
        return np.array(self.data), np.array(self.weights), np.array(self.label)

    def __getitem__(self, index):
        # Data, Weights, Label
        return np.array(self.data.iloc[index], dtype=np.float32), np.array(self.weights.iloc[index], dtype=np.float32), self.label.iloc[index]

    def __len__(self):
        return self.n_samples
