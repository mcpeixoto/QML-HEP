"""
Author: Miguel Caçador Peixoto
Description: 
    For a given set of HPs, this script will train a QML model,
a SVM and a Logistic Regression model.
"""

# Imports
from tqdm import tqdm
import os
from os.path import join
from datetime import datetime
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from pennylane import numpy as np

from utils.helper import GridSearch, NestablePool, get_random_numbers
from qmlhep.qml import AdamModel, OptunaModel
from qmlhep.config import analisys_results_path, hp_space, N_PROCESSES


def svm_analisys(hp):
    clf = svm.SVC(kernel="rbf", probability=True)

    trainer = globals()[hp["study_name"]](**hp, load_data=False)
    trainer.normalization = "ML"
    trainer.load_datasets()

    X_train, W_train, Y_train = trainer.train_dataset.all_data()

    # Retormalize weights
    W_train[Y_train == 1] = (W_train[Y_train == 1] / W_train[Y_train == 1].sum()) * W_train.shape[0] / 2
    W_train[Y_train == 0] = (W_train[Y_train == 0] / W_train[Y_train == 0].sum()) * W_train.shape[0] / 2

    # Train SMV
    clf.fit(X_train, Y_train, sample_weight=W_train)

    del X_train, W_train, Y_train

    #### Validation data ####
    X_val, W_val, Y_val = trainer.val_dataset.all_data()

    # Retormalize weights
    W_val[Y_val == 1] = (W_val[Y_val == 1] / W_val[Y_val == 1].sum()) * W_val.shape[0] / 2
    W_val[Y_val == 0] = (W_val[Y_val == 0] / W_val[Y_val == 0].sum()) * W_val.shape[0] / 2

    # Predict
    y_val_scores = clf.predict_proba(X_val)
    y_val_scores = y_val_scores[:, 1]

    # Add everything into hp
    hp["X_val"] = X_val
    hp["Y_val"] = Y_val
    hp["W_val"] = W_val
    hp["Y_val_scores"] = y_val_scores

    #### Test data ####
    X_test, W_test, Y_test = trainer.test_dataset.all_data()

    # Retormalize weights
    W_test[Y_test == 1] = (W_test[Y_test == 1] / W_test[Y_test == 1].sum()) * W_test.shape[0] / 2
    W_test[Y_test == 0] = (W_test[Y_test == 0] / W_test[Y_test == 0].sum()) * W_test.shape[0] / 2

    # Predict
    y_test_scores = clf.predict_proba(X_test)
    y_test_scores = y_test_scores[:, 1]

    # Add everything into hp
    hp["X_test"] = X_test
    hp["Y_test"] = Y_test
    hp["W_test"] = W_test
    hp["Y_test_scores"] = y_test_scores

    return hp


def logistic_regression_analisys(hp):
    clf = LogisticRegression()

    # Training data
    trainer = globals()[hp["study_name"]](**hp, load_data=False)
    trainer.normalization = "ML"
    trainer.load_datasets()

    X_train, W_train, Y_train = trainer.train_dataset.all_data()
    # Retormalize weights
    W_train[Y_train == 1] = (W_train[Y_train == 1] / W_train[Y_train == 1].sum()) * W_train.shape[0] / 2
    W_train[Y_train == 0] = (W_train[Y_train == 0] / W_train[Y_train == 0].sum()) * W_train.shape[0] / 2

    # Train SMV
    clf.fit(X_train, Y_train, sample_weight=W_train)

    del X_train, W_train, Y_train

    # Validation data
    X_val, W_val, Y_val = trainer.val_dataset.all_data()

    # Retormalize weights
    W_val[Y_val == 1] = (W_val[Y_val == 1] / W_val[Y_val == 1].sum()) * W_val.shape[0] / 2
    W_val[Y_val == 0] = (W_val[Y_val == 0] / W_val[Y_val == 0].sum()) * W_val.shape[0] / 2

    # Predict
    y_val_scores = clf.predict_proba(X_val)
    y_val_scores = y_val_scores[:, 1]

    # Add everything into hp
    hp["X_val"] = X_val
    hp["Y_val"] = Y_val
    hp["W_val"] = W_val
    hp["Y_val_scores"] = y_val_scores

    #### Test data ####
    X_test, W_test, Y_test = trainer.test_dataset.all_data()

    # Retormalize weights
    W_test[Y_test == 1] = (W_test[Y_test == 1] / W_test[Y_test == 1].sum()) * W_test.shape[0] / 2
    W_test[Y_test == 0] = (W_test[Y_test == 0] / W_test[Y_test == 0].sum()) * W_test.shape[0] / 2

    # Predict
    y_test_scores = clf.predict_proba(X_test)
    y_test_scores = y_test_scores[:, 1]

    # Add everything into hp
    hp["X_test"] = X_test
    hp["Y_test"] = Y_test
    hp["W_test"] = W_test
    hp["Y_test_scores"] = y_test_scores
    return hp


def qml_analisys(hp):
    hp["name"] = f'{hp["name"]}_{hp["random_seed"]}'

    # Get model for training
    trainer = globals()[hp["study_name"]](**hp)

    # Train QML
    trainer.train()

    # Validation data
    X_val, W_val, Y_val = trainer.val_dataset.all_data()

    X_val = np.array(X_val, requires_grad=False)
    Y_val = np.array(Y_val, requires_grad=False)
    W_val = np.array(W_val, requires_grad=False)

    # Retormalize weights
    W_val[Y_val == 1] = (W_val[Y_val == 1] / W_val[Y_val == 1].sum()) * W_val.shape[0] / 2
    W_val[Y_val == 0] = (W_val[Y_val == 0] / W_val[Y_val == 0].sum()) * W_val.shape[0] / 2

    # Load weights
    weights = trainer.load_model()

    # Compute predictions
    y_val_scores = np.array([trainer.classifier(weights, x) for x in X_val])
    y_val_scores = (y_val_scores + 1) / 2

    # Add everything into hp
    hp["X_val"] = X_val
    hp["Y_val"] = Y_val
    hp["W_val"] = W_val
    hp["Y_val_scores"] = y_val_scores

    #### Test data ####
    X_test, W_test, Y_test = trainer.test_dataset.all_data()

    # Retormalize weights
    W_test[Y_test == 1] = (W_test[Y_test == 1] / W_test[Y_test == 1].sum()) * W_test.shape[0] / 2
    W_test[Y_test == 0] = (W_test[Y_test == 0] / W_test[Y_test == 0].sum()) * W_test.shape[0] / 2

    #  Compute predictions
    y_test_scores = np.array([trainer.classifier(weights, x) for x in X_test])
    y_test_scores = (y_test_scores + 1) / 2

    # Add everything into hp
    hp["X_test"] = X_test
    hp["Y_test"] = Y_test
    hp["W_test"] = W_test
    hp["Y_test_scores"] = y_test_scores
    return hp


def analisys_experiment(work_load):
    hp, n_runs = work_load

    save_path = join(analisys_results_path, hp["study_name"], f'{hp["name"]}.pkl')

    # Check if work_load already done
    if os.path.exists(save_path):
        print(f'[+] Workload {hp["study_name"]}-{hp["name"]} already done')
        return
    else:
        print(f'[+] Starting {hp["study_name"]}-{hp["name"]} | #runs: {n_runs}  ..')

    # Multiprocessing
    pool = NestablePool(processes=n_runs)

    # Get seeds
    random_seeds = get_random_numbers(n_runs)

    # Get diferent HPs
    HPs = [dict(hp, random_seed=seed) for seed in random_seeds]
    assert len(HPs) == n_runs

    # Train the diferent models
    qml_world = [pool.apply_async(qml_analisys, (hp,)) for hp in HPs]
    svm_world = [pool.apply_async(svm_analisys, (hp,)) for hp in HPs]
    log_world = [pool.apply_async(logistic_regression_analisys, (hp,)) for hp in HPs]
    pool.close()

    # Get results
    qml_world = [w.get() for w in qml_world]
    svm_world = [w.get() for w in svm_world]
    log_world = [w.get() for w in log_world]

    pool.join()

    # Save values
    with open(save_path, "wb") as f:
        pickle.dump([qml_world, svm_world, log_world], f)

    return


if __name__ == "__main__":

    ###################
    ## CONFIGURATION ##
    ###################

    # Number of random samplings for each HP set
    n_runs = 5

    ###################

    N_PROCESSES = int(N_PROCESSES / n_runs)

    if N_PROCESSES <= 0:
        print("WARNING: Number of runs is higher than number of cores. This process will be extremely slow.")
        N_PROCESSES == 1
    else:
        print(f'[+] Using {N_PROCESSES} cores')

    # Print current time
    print("> STARTED:", start_time := datetime.now())

    for study_name in ["AdamModel", "OptunaModel"]:
        save_dir = join(analisys_results_path, study_name)
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                pass

        # Prints
        print("-" * 10)
        print("> Study name:", study_name)
        print("> Number of available processes:", N_PROCESSES)
        print("> Number of runs per process:", n_runs)
        print("> Number of available workers:", n_runs * N_PROCESSES)

        # Get all the combinations of HP space
        hp_space["study_name"] = study_name

        # Multiprocessing start
        work_load = [(hp, n_runs) for hp in GridSearch(hp_space)]

        print("> Total number of processes to be completed:", len(work_load))
        print("-" * 10)

        with NestablePool(processes=N_PROCESSES) as p:
            r = list(tqdm(p.imap(analisys_experiment, work_load), total=len(work_load), desc=f"[{study_name}] Waiting for processes to finish..."))

    # Print current time
    print("> ENDED:", datetime.now())
    print(f"Total time elapsed: {datetime.now() - start_time}")
