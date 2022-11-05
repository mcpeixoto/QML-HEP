"""
Author: M. Gabriela Oliveira
Description:
    This script is used to find the reduced dataset using the KMeans algorithm.
After the use of the KMeans algorithm, the nearest points of each centroid are found.
The nearest points are used to create the reduced dataset.
"""


import os

os.environ["OMP_NUM_THREADS"] = "25"
from tqdm import tqdm
import pickle
from multiprocessing import Pool
import pickle
from sklearn.cluster import KMeans
import warnings
import numpy as np
import pandas as pd
from os.path import join

from qmlhep.data_handling.dataset import ParticlePhysics
from qmlhep.config import others_path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

# Defining random seed
random_seed = 42

# Load data
train_data = ParticlePhysics(category="train", standardization="ML", random_seed=random_seed).all_data_Dataframe()
train_data.drop(columns=["name"], inplace=True)

print("Starting KMeans")
print("Part 1 started")
############################################### Part 1 ###############################################
"""
This part is used to find the centroids of the train and validation/test data.
At the end, the centroids are saved in a pandas Dataframe.
"""
# OPENMP NUM THREADS


def Kmeans_work(n_clusters, random):
    "This function is used to parallelize the use of the KMeans algorithm"
    "n_clusters: number of clusters"
    "random: random seed"

    # To save the results
    dic = {}

    # Train data
    features = train_data.columns[:-2]
    X_train, Y_train, W_train = train_data[features], train_data["label"], train_data["weights"]

    # Work with background and signal separately
    X_train_0 = X_train[Y_train == 0]
    X_train_1 = X_train[Y_train == 1]
    Y_train_0 = Y_train[Y_train == 0]
    Y_train_1 = Y_train[Y_train == 1]

    # Initialize the KMeans
    kmeans_0 = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random)
    kmeans_1 = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random)

    # Fit to find the centroids
    kmeans_0.fit(X_train_0, Y_train_0, sample_weight=W_train[Y_train == 0])
    kmeans_1.fit(X_train_1, Y_train_1, sample_weight=W_train[Y_train == 1])

    # Centroids
    X_cluster_0 = kmeans_0.cluster_centers_
    X_cluster_1 = kmeans_1.cluster_centers_
    X_cluster = np.concatenate((X_cluster_0, X_cluster_1), axis=0)

    # Save information
    dic["#Clusters"] = n_clusters
    dic["Centrus"] = X_cluster
    dic["Centrus0"] = X_cluster_0
    dic["Centrus1"] = X_cluster_1

    return dic


# Fixed #Processes
# If we want all CPUs ->  os.cpu_count()
N_PROCESSES = 6

# Create a pool of processes
pool = Pool(N_PROCESSES)

processes = []
# Number of clusters
for n_clusters in [50, 250, 500, 2500]:
    p = pool.apply_async(Kmeans_work, args=(n_clusters, 42))
    processes.append(p)

print("[+] Total number of processes to be completed:", len(processes))

# Wait for all processes to finish TODO: Improve tqdm
for p in tqdm(processes, total=len(processes), desc="Waiting for processes to finish..."):
    p.wait()

# Close the pool
pool.close()

# Compile results
results = [p.get() for p in processes]

centroid_info = pd.DataFrame.from_dict(results)

print("[+] Centroids found")
print("[+] Part 1 finished")

print("Part 2 started")
################################################### Part 2 ###################################################
"""
This part is used to find the n-nearest neighbors of the centroids.
At the end, the n-nearest neighbors are saved in pandas Dataframe.
"""

# Define the number of neighbors
n_neighbors = 10

# Function to found the nearest points of each centroid
def neighbors_work(X, Y, W, centroids, elem):
    """
    X: Dataframe with the features - train, validation or test
    Y: Dataframe with the labels - train, validation or test
    W: Dataframe with the weights - train, validation or test
    centroids: Dataframe with the centroids (found by KMeans)
    elem: number of centroid
    """
    # Dictionary to save the neighbors
    dic = {}

    # Auxiliar array to save the distances
    dist = []

    # calculate the distance for each point
    for i in range(len(X)):
        diff = X.iloc[i] - centroids.iloc[0][0][elem]
        val = np.sum(diff**2)
        dist.append(val)

    # index sort
    dist = np.argsort(dist)

    # Save the nearest points
    for j in range(n_neighbors):
        tmp = {}
        tmp["min_x"] = X.iloc[dist[j]]
        tmp["min_y"] = Y[dist[j]]
        tmp["min_w"] = W[dist[j]]

        # add the point to the dictionary
        dic[f"neigh_{j}"] = tmp

    return dic


# auxiliar function to found the nearts points of centroids
def resampler(X, Y, W, centroids):

    X.reset_index(drop=True, inplace=True)
    Y.reset_index(drop=True, inplace=True)
    W.reset_index(drop=True, inplace=True)

    # Fixed #Processes
    # If we want all CPUs ->  os.cpu_count()
    N_PROCESSES = 50

    # Create a pool of processes
    pool = Pool(N_PROCESSES)

    processes = []

    # found the nearest point for each centroid
    # len(centroids.iloc[0][0]) = number of centrus
    for elem in range(len(centroids.iloc[0][0])):
        p = pool.apply_async(neighbors_work, args=(X, Y, W, centroids, elem))
        processes.append(p)

    print("[+] Total number of processes to be completed:", len(processes))

    # Wait for all processes to finish
    for p in tqdm(processes, total=len(processes), desc="Waiting for processes to finish..."):
        p.wait()

    # Close the pool
    pool.close()

    # Compile results
    results = [p.get() for p in processes]

    return results


infos = []
for n_centrus in [50, 250, 500, 2500]:

    resamp = {}
    features = train_data.columns[:-2]
    X_train, Y_train, W_train = train_data[features], train_data["label"], train_data["weights"]

    # #centrus
    centrus_k = centroid_info["#Clusters"] == n_centrus
    centrus_k = centroid_info[centrus_k]

    # found the nearests points for each centroid
    Resampled_0 = resampler(X_train[Y_train == 0], Y_train[Y_train == 0], W_train[Y_train == 0], pd.DataFrame(centrus_k["Centrus0"]))
    Resampled_1 = resampler(X_train[Y_train == 1], Y_train[Y_train == 1], W_train[Y_train == 1], pd.DataFrame(centrus_k["Centrus1"]))

    # infos
    resamp["#Clusters"] = n_centrus
    resamp["Resampled0"] = Resampled_0
    resamp["Resampled1"] = Resampled_1
    infos.append(resamp)

    centrus_infos = pd.DataFrame.from_dict(infos)
print("[+] Resampling finished")
print("[+] Part 2 finished")

print("Part 3 started")
################################################### Part 3 ###################################################
"""
This part is used to compute the weighted average of the n-nearest neighbors of the centroids.
At the end, the reduced dataset is saved in pandas Dataframe and in a pickle file.
"""

# Load data
features = train_data.columns[:-2]
X_train, Y_train, W_train = train_data[features], train_data["label"], train_data["weights"]


# Determine the new weights for the training and validation set
# based on the number of samples
len_1_train = len(X_train[Y_train == 1])
len_0_train = len(X_train[Y_train == 0])

weight_1_train = 1 / len_1_train
weight_0_train = 1 / len_0_train

# List to store the results
results = []
for n_clusters in [50, 250, 500, 2500]:
    tmp = {}

    # n selected clusters
    infos_aux = centrus_infos["#Clusters"] == n_clusters

    # new dataframe with only the selected #Clusters
    infos_aux = centrus_infos[infos_aux]

    tmp["#Clusters"] = n_clusters

    # Lists to store the results/datasets
    X_train = []
    Y_train = []
    W_train = []

    # Concatenate the signal and background datasets
    resampler_test = np.concatenate([infos_aux["Resampled0"].iloc[0], infos_aux["Resampled1"].iloc[0]])

    # Weighted average of the neighbors
    for i in range(2 * n_clusters):

        # Train
        resampled = resampler_test[i]
        x = 0
        w_m = 0

        for j in list(resampled):
            x += resampled[j]["min_x"] * resampled[j]["min_w"]
            w_m += resampled[j]["min_w"]

        X_train.append(x / w_m)

        if resampled["neigh_0"]["min_y"] == 1:
            W_train.append(weight_1_train)
        else:
            W_train.append(weight_0_train)

        Y_train.append(resampled["neigh_0"]["min_y"])

    tmp["X_train"] = X_train
    tmp["Y_train"] = Y_train
    tmp["W_train1"] = W_train

    results.append(tmp)

    # KMeans dataset for dataset size reduction study
    with open(join(others_path, "kmeans_dataset_train.pkl"), "wb") as f:
        pickle.dump(results, f)

print("[+] Part 3 finished")
print("[+] KMeans dataset for dataset size reduction study saved")
