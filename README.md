# ELEC0135: AMLS II 2023-2024 Assignment

## Project Overview

The author chose to tackle the Cassava Leaf Disease Classification competition on Kaggle: https://www.kaggle.com/competitions/cassava-leaf-disease-classification

This repository contains all the code required to download & preprocess the data, along with training the deep learning networks.

## Repository Contents

`/data_processing/`
> `cld_data_download.ipynb`: This interactive notebook downloads the data from Kaggle, reduces the dataset to just the jpg files and labels, and then uploads it to the author's Google Drive account for convenience.

>` pre-processing.py`: This file contains the code to download the adjusted dataset from Google Drive and make it available to the machine learning algorithms in standard formats (i.e. TensorFlow Datasets).  It also contains utility methods to cache, augment, and up-sample the datasets.

`/model/`
> `util.py`: This file has the functions required to create the convolutional neural network (ConvNet) models, along with the utility code to run and store details of the tasks.  Also included here are some ancilliary functions to produce confusion matrices and learning rate curve graphs.

`/results/`: The directory hosts the results.  Each task has its own subdirectory, which contains the confusion matrices and learning rate curves.  The parent directory contains the full training history (`train_details.csv`) and test scores (`test_scores.csv`).

`/workings/`
> `assignment-workings.ipynb`: As noted in the 'Development Approach' section of the report, the author used this notebook to develop the solutions on Kaggle before carving the code into the module structure described here.  It isn't meant to be part of the assignment submission, per se, but the revision history evidences how the solution was built-up.

## Requirements

## Running the Code


