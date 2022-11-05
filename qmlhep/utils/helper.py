"""
Author: Miguel Caçador Peixoto
Description: 
    Script containing helper functions for the qmlhep package.
"""

# Imports
import hashlib
from pennylane import numpy as np
import pickle
import itertools
from os.path import join, basename, exists
import numpy
import pandas as pd
import multiprocessing.pool

from qmlhep.config import others_path

#########################################################
#######  Helper Functions for the QMLHEP package  #######
#########################################################

####################################
### MD5 Hashing & File Integrity ###
####################################

def get_md5(fpath):
    """
    Compute the md5 checksum of a file.
    """
    hash_md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_integrity(fpath, lookuptable):
    """
    Check the integrity of a file using its md5 hash
    by comparing it to the lookup table.
    """
    md5_returned = get_md5(fpath)
    md5_expected = lookuptable[basename(fpath).replace(".tmp", "")]
    return md5_returned == md5_expected


###################################
#             Metrics             #
###################################

def square_loss(labels, predictions):
    loss = []
    for l, p in zip(labels, predictions):
        loss.append((l - p) ** 2)
    loss = np.array(loss, requires_grad=True)
    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


###################################
# Random Number Generator Wrapper #
###################################

def get_random_numbers(n):
    """
    Get n random numbers
    """
    seeds_path_persistant = join(others_path, "seeds.npy")

    # Set random seed for reproducibility
    numpy.random.seed(42)

    # Check if seeds.npy exists
    if not exists(seeds_path_persistant):
        # Create seeds.npy
        seeds = numpy.random.randint(0, 2**32, n)
        numpy.save(seeds_path_persistant, seeds)

    else:
        # Load seeds.npy
        seeds = numpy.load(seeds_path_persistant)

        if len(seeds) < n:
            # Create more seeds and append
            seeds = numpy.concatenate((seeds, numpy.random.randint(0, 2**32, n - len(seeds))))
            numpy.save(seeds_path_persistant, seeds)

    return seeds[:n]


###########################################
# Overwritting default MultiProcess Class #
###########################################

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


###########################################
#        Other Relevant Functions         #
###########################################

def load_SBS_book():
    """
    Load the SBS book
    """
    book_path = join(others_path, "SBS.pkl")

    if not exists(book_path):
        raise Exception(f"[!] File {book_path} not found. Please run model_select script first.")

    with open(book_path, "rb") as f:
        book = pickle.load(f)

    return book


def get_features(n_features):
    """
    Given a number of features, return a list of features
    selected by the SBS algorithm.
    """

    book = load_SBS_book()

    features = list(set(book[n_features]))
    assert len(features) == n_features, "[!] Invalid number of features"

    return features


def join_dicts(dict1, dict2):
    """
    Join two dictionaries together
    """
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].append(value)
        else:
            dict1[key] = value
    return dict1


def GridSearch(params, fixed=None):
    """
    Given a dictionary of hyperparameters, return a
    dictionary iterator of hyperparameters.

    Example:
    params = {'n_layers': [1, 2, 3], 'n_qubits': [2, 3, 4], 'n_features': [2, 3, 4]}
    params_iterator = GridSearch(params)
    for x in params_iterator:
        print(x)
    """

    # Set all values to list
    for key, value in params.items():
        if not isinstance(value, list):
            params[key] = [value]

    # Get all keys
    keys = list(params.keys())

    # Get all values
    values = list(params.values())

    # Get all combinations
    combinations = list(itertools.product(*values))

    # Create dictionary iterator
    buff = []
    for i, combination in enumerate(combinations):
        book = dict(zip(keys, combination))
        if fixed is not None:
            book = join_dicts(book, fixed)
        book["name"] = str(i)
        buff.append(book)

    return buff


######################
# Format KMeans Data #
######################

def get_kmeans_data(n_datapoints):
    """
    Author: M. Gabriela Jordão
    """

    # Full dataset from Kmeans
    with open(join(others_path, "kmeans_dataset_train.pkl"), "rb") as f:
        samples = pickle.load(f)
    samples = pd.DataFrame.from_dict(samples)

    # Choose the number of points
    samples_aux = samples["#Clusters"] == n_datapoints / 2
    samples_aux = samples[samples_aux]
    X = pd.DataFrame(samples_aux["X_train"].iloc[0])
    Y = pd.DataFrame(samples_aux["Y_train"].iloc[0], columns=["label"], index=X.index)
    W = pd.DataFrame(samples_aux["W_train1"].iloc[0], columns=["weights"], index=X.index)

    return pd.concat([X, Y, W], axis=1)
