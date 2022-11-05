"""
Author: Miguel Caçador Peixoto
Description: 
    Script containing the pipeline for infering the best model on a real quantum computer, provided by IBM.
"""

# Imports
from tqdm import tqdm
import os
from datetime import datetime
from os.path import basename, join
from qiskit import *
import pickle
from pennylane_qiskit import IBMQDevice
import time
from multiprocess import Pool
from multiprocessing import Manager
# Remove warnings
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("urllib3").setLevel(logging.ERROR) # Remove HTTP warnings

from qmlhep.qml import AdamModel, OptunaModel
from qmlhep.utils.plot_results import Painter
from qmlhep.config import qc_results_path, ibm_systems
from utils.helper import NestablePool

# Load IBM provider
IBMQ.load_account()
provider = IBMQ.providers()[0]
provider.backends()

##############################
#### Start of the script
##############################

class Run_On_Quantum_Computer:
    def __init__(self, ball):
        self.model, self.use_test_data, self.backend_name, self.position, self.lock = ball
        self.position += 1

        self.painter = Painter(use_test_data=use_test_data)
        self.best_name, self.best_run, (self.qml_worlds, self.svm_worlds, self.log_worlds) = self.painter.get_best_name(self.model)

        # For every undersampling of the best run, run on the quantum computer
        with self.lock:
            bar = tqdm(total=len(self.qml_worlds), desc=f"Running on {self.backend_name}", leave=False, position=self.position*3, dynamic_ncols=True)
            time.sleep(0.5) # To make sure we add an delay between processes (So IBM don't get mad)

        for qml_world in self.qml_worlds:
            self.distributer(qml_world)
            with self.lock:
                bar.update(1)

        with lock:
            bar.close()

    def run(self, ball):
        """
        This will take a single event, initialize the circuit, run it on the quantum computer and return the result
        """
        single_batch, hp = ball
        hp['load_data'] = False

        result = None
        while result is None:
            # Sometimes the IBM's API fails, so we need to try again
            try:                
                trainer = globals()[self.model](**hp)

                trainer.dev = IBMQDevice(backend=self.backend_name, wires=trainer.n_features, provider=provider, shots=20000)
                trainer.dev.set_transpile_args(optimization_level=3)

                weights = trainer.load_model()
                result = trainer.classifier(weights, single_batch)

            # HTTP Error, try again
            except:
                time.sleep(2)

        result = float(result + 1) / 2

        return result

    def distributer(self, qml_world):
        """
        This function will initialize the pool of workers and distribute the events to them
        This is done because we can instanciate a maximum of 5 jobs in a queue for a single quantum computer
        """
        if use_test_data:
            self.y_scores_key = "Y_test_scores"
            self.x_data_key = "X_test"
            self.y_data_key = "Y_test"
            self.w_data_key = "W_test"
            self.save_path_extra = "test"
        else:
            self.y_scores_key = "Y_val_scores"
            self.x_data_key = "X_val"
            self.y_data_key = "Y_val"
            self.w_data_key = "W_val"
            self.save_path_extra = "val"

        X = qml_world[self.x_data_key]
        name = qml_world["name"]
        save_path = join(qc_results_path, f"{self.backend_name}_{name}_{self.save_path_extra}.pkl")

        if os.path.exists(save_path):
            return

        work = [(single_batch, qml_world) for single_batch in X]

        # Can't be len(work) because we're limited by IBM - Max of 5 jobs in Queue for 1 device
        pool = Pool(5)
        results = list(tqdm(pool.imap(self.run, work), total=len(work), desc=f"> Current Run {name}", leave=True, position=self.position*3+1, dynamic_ncols=True))

        pool.close()
        pool.join()

        with open(save_path, "wb") as f:
            pickle.dump(results, f)

        return results


##############################
# Main
##############################

if __name__ == "__main__":

    ##############################
    #### Configure
    ##############################

    model = "OptunaModel"
    use_test_data = True

    # Print information to user
    print(f"Running the best hyperparameter-set of *{model}* on a real quantum systems")
    print(f"Using test data: {use_test_data}")
    print("Will run on the following systems:")
    for system in ibm_systems:
        print(f"- {system}")

    ##############################
    #### Run
    ##############################

    # Parallelize runing on quantum computers
    start_time = datetime.now()
    print(f">>>> Start time: {start_time}")

    pool = NestablePool(len(ibm_systems))
    lock = Manager().Lock()     # To prevent multiple processes writing to stdout at the same time.
    pool.map(Run_On_Quantum_Computer, [(model, use_test_data, backend_name, position, lock) for position, backend_name in enumerate(ibm_systems)])

    print(f">>>> End time: {datetime.now()} - Total time elapsed: {datetime.now() - start_time}\n")
