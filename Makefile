# Makefile for the QML-HEP python package
# Author: Miguel CaΓ§ador Peixoto

# Help message
help:
	@echo "Makefile for the QML-HEP python package"
	@echo ""
	@echo "Usage:"
	@echo "    make <target>"
	@echo ""
	@echo "Targets:"
	@echo "    help                 π¬ This help message"
	@echo "    install              π¦ Install dependencies, download and pre-process data."
	@echo "    download             π₯ Download data from the web."
	@echo "    preprocess           π Pre-process data."
	@echo "    run_sbs              π Run the feature selection SBS algorithm."
	@echo "    run_kmeans           πββοΈ Generate the KMeans algorithm dataset."
	@echo "    run_gridsearch       ποΈββοΈ Train VQCs, SVMs and LR based on the gridsearch specified in config."
	@echo "    run_qc               π¨βπ» Infer the best-performing VQC on IBM's quantum computers. "
	@echo "    run_best_kmeans      ποΈββοΈ Train the 3 models of the best-performing HP set on the kmeans dataset and infer performance on the test dataset."
	@echo ""

# Variables
CURRENT_DIR = $(shell pwd)
SHELL = /bin/bash

install:
	@echo "Creating virtual enviroment..."
	python -m venv .env
	source .env/bin/activate
	@echo "Installing dependencies..."
	.env/bin/python -m pip install -r requirements.txt
	.env/bin/python -m pip install -e .
	make download

download:
	@echo "Downloading data from the web..."
	source .env/bin/activate
	.env/bin/python qmlhep/data_handling/download.py
	@echo "Done."

preprocess:
	@echo "Pre-processing data..."
	source .env/bin/activate
	.env/bin/python qmlhep/data_handling/preprocess.py
	@echo "Done."

run_sbs:
	@echo "Running SBS algorithm..."
	source .env/bin/activate
	.env/bin/python qmlhep/data_handling/sbs.py
	@echo "Done."

run_kmeans:
	@echo "Running KMeans algorithm..."
	source .env/bin/activate
	.env/bin/python qmlhep/data_handling/kmeans.py
	@echo "Done."

run_gridsearch:
	@echo "Running Gridsearch..."
	source .env/bin/activate
	.env/bin/python qmlhep/run_gridsearch.py
	@echo "Done."

run_qc:
	@echo "Running QC..."
	source .env/bin/activate
	.env/bin/python qmlhep/run_qc.py
	@echo "Done."

run_best_kmeans:
	@echo "Running Best Kmeans..."
	source .env/bin/activate
	.env/bin/python qmlhep/run_best_kmeans.py
	@echo "Done."
