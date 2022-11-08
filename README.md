<picture>
    <source media="(prefers-color-scheme: dark)" srcset=".media/qml-hep-white.png">
    <source media="(prefers-color-scheme: light)" srcset=".media/qml-hep-black.png">
    <img src=".media/qml-hep-white.png">
</picture>
  
# [Fitting a Collider in a Quantum Computer: Tackling the Challenges of Quantum Machine Learning for Big Datasets](https://arxiv.org/abs/2211.03233)

Authors:
Miguel Caçador Peixoto (1), Nuno Filipe Castro (1,2), Miguel Crispim Romão (1,3), Maria Gabriela Jordão Oliveira (1), Inês Ochoa (4)

**1** - LIP - Laboratório de Instrumentação e Física Experimental de Partículas, Escola de Ciências, Campus de Gualtar, Universidade do Minho, 
4701-057 Braga, Portugal
 
**2** - Departamento de Física, Escola de Ciências, Campus de Gualtar, Universidade do Minho, 
4701-057 Braga, Portugal

**3** - Department of Physics and Astronomy, University of Southampton,
SO17 1BJ Southampton, United Kingdom

**4** - LIP - Laboratório de Instrumentação e Física Experimental de Partículas, 
Av. Prof. Gama Pinto, 2, 1649-003 Lisboa, Portugal


## Abstract

The current quantum systems have significant limitations affecting the processing of large datasets and high dimensionality typical of high energy physics. In the current work, feature and data prototype selection techniques were studied within this context. A grid search was performed and quantum machine learning models were trained and benchmarked against classic shallow machine learning methods, trained both in the reduced and the complete datasets. The performance of the quantum algorithms was found to be comparable to the classic ones, even when using large datasets.

## Install


### 1. Pre-requisites

**Software**

- Unix-based operating system (Linux)
- [Python 3.8+](https://www.python.org/downloads/)

**Hardware**

Hardware requirements may vary, depending on the amount of parallelism your machine is capable of.

For our case, this code ran on a server with 112 threads, which meant that during training 112 QC emulations were being run at a single time (Pennylane's 'default.qubit' simulator is single-core locked). With this workload, around 50Gb of RAM were being used.

### 2. Makefile

A standard GNU Make file is provided to help you get things set up locally.

The Makefile has these commands available and can be accessed by ```make <COMMAND>```.

```text
help                	💬 This help message
install             	📦 Install dependencies and download data.
download            	📥 Download data from the web.
preprocess          	📈 Pre-process data.
run_sbs             	🏃 Run the feature selection SBS algorithm.
run_kmeans          	🏃‍♀️ Generate the KMeans algorithm dataset.
run_gridsearch      	🏋️‍♀️ Train VQCs, SVMs, and LR based on the grid search specified in config.
run_qc              	👨‍💻 Infer the best-performing VQC on IBM's quantum computers.
run_best_kmeans     	🏋️‍♀️ Train the 3 models of the best-performing HP set on the KMeans dataset and infer performance on the test dataset.
```

So for getting things set up just run:


```
make install
```

This should install everything automatically. More specifically, it will create a python environment, install all the necessary modules and download the dataset.  

## Running

After the installation process is complete, it's possible to use simple *makefile* commands to reproduce the results shown on the paper.

### Preparing The Data

Before running anything, it's necessary to prepare the data for the next steps.

First, it's necessary to preprocess the downloaded data which includes applying data cuts, preprocessing Montecarlo weights, and removing unnecessary columns.

```
make preprocess
```

After that, it's possible to run the SBS algorithm to select the best features for the dataset and save the results.


```
make run_sbs
```

And if we want to run the performance study on the KMeans dataset in the future, its generation is also required:

```
make run_kmeans
```

### Training & Inference

This section is divided into 3 parts, the first one is the grid search, which is used to find the best set of hyperparameters for the VQC (and consequently, the shallow methods). After that, inference on the best-performing VQC is performed on IBM's quantum computers. And finally, the best-performing VQC and shallow ML models are trained on the KMeans dataset and the performance compared to random subsampling.

For the grid search, whose HP space is specified in *config.py*, it's possible to run the following command:

```
make train_gridsearch
```

Note that by default all CPU cores are used for training. If you wish to use a different number of cores, please change the ```N_PROCESSES``` parameter in *config.py*.

After the grid search is complete, it's possible to run inference on the test dataset of the best-performing VQC on IBM's quantum computers. For this, it's necessary to have an IBMQ account and have the API token configurated. For this purpose, please uncomment the code for "IBM Configuration" in *config.py* and insert your key. If you don't have an account, you can create one [here](https://quantum-computing.ibm.com/).

For running inference on the best-performing VQC on quantum computers, run the following command:

```
make run_qc
```

Finally, if we want to train the best-performing VQC and shallow ML models on the KMeans dataset and compare the performance to random subsampling, we can run the following command:

```
make train_best_kmeans
```

### Results Visualization

For visualizing the results, multiple notebooks are provided in the *notebooks* folder. They are: *results_gridsearch.ipynb*, *results_qc.ipynb* and *results_kmeans.ipynb*. They are self-explanatory and should be easy to follow.

### Additional Notebooks

In the *notebooks* folder, there are also notebooks for the following purposes:
- *data_exploration.ipynb*: For exploring the preprocessed data.
- *baseline.ipynb*: Measuring the baseline performance of an XGBClassifier on the full dataset, and other performance metrics PCA and SBS related.
- *kmeans_dataset.ipynb*: Performance analysis of the KMeans dataset using logistic regression

Note: Make sure you are using the same python environment as the one used for the installation process when executing the notebooks, this is located in the *.env* folder (created after instaling).

Note 2: If you desire to use GPU on the supported models, please change the ```use_gpu``` parameter in *config.py* to ```'True'```.

## File Structure

- qml
    -  `base.py` - Base class for training and evaluating models
    -  `adam.py` - Adam optimizer version of the base class
    -  `optuna.py` - Optuna optimizer version of the base class
- data_handling
    -  `download.py` - Download data from the web
    -  `preprecess.py` - Preprocessing of downloaded data
    -  `dataset.py` - The main dataset class
    -  `sbs.py` - Feature selection (SBS algorithm)
    -  `kmeans.py`- Dataset size reduction (KMeans Algorithm)
- utils
    -  `helper.py` - file containing helper functions
    -  `plot_results.py` - file containing functions for plotting results
- notebooks - Jupyter notebooks for plotting
- data* - the datasets used in the paper
    - processed - the pre-processed datasets
    - raw - the original datasets
    - other - other persistent data needed for the project such as NumPy seeds and SBS data.
- results* - the results of the experiments
- plots* - the plots generated output
- examples
-  `config.py` - the configuration files
-  `run_gridsearch.py` - the main file for running the grid-search
-  `run_qc.py` - the main file for running inference on the best-performing VQC on IBM's quantum computers
-  `run_best_kmeans.py` - the main file for training the best-performing VQC on the KMeans dataset and comparing its performance with the shallow methods on the test dataset
-  `Makefile`

`*` - Automatically generated folders

# Reference

- [Fitting a Collider in a Quantum Computer: Tackling the Challenges of Quantum Machine Learning for Big Datasets, arxiv.org](https://arxiv.org/abs/2211.03233)
