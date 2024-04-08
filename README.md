# ELEC0135: AMLS II 2023-2024 Assignment

## Project Overview

The author chose to tackle the Cassava Leaf Disease Classification competition on Kaggle: https://www.kaggle.com/competitions/cassava-leaf-disease-classification

This repository contains all the code required to download & pre-process the data, along with training the deep learning networks.

## Repository Contents

`/data_processing/`
> `cld_data_download.ipynb`: This interactive notebook downloads the data from Kaggle, reduces the dataset to just the jpg files and labels, and then uploads it to the author's Google Drive account for convenience (public access).

>` pre-processing.py`: This file contains the code to download the adjusted dataset from Google Drive and make it available to the machine learning algorithms in standard formats (i.e. TensorFlow Datasets).  It also contains utility methods to cache, augment, and up-sample the datasets.

`/model/`
> `util.py`: This file has the functions required to create the convolutional neural network (ConvNet) models, along with the utility code to run and store details of the tasks.  Also included here are some ancilliary functions to produce confusion matrices and learning rate curve graphs.

`/results/`:
> The directory hosts the results.  Each task has its own subdirectory, which contains the confusion matrices and learning rate curves.  The parent directory contains the full training history (`train_details.csv`) and test scores (`test_scores.csv`).

`/workings/`
> `assignment-workings.ipynb`: As noted in the 'Development Approach' section of the report, the author used this notebook to develop the solutions on Kaggle before carving the code into the module structure described here.  It isn't meant to be part of the assignment submission, per se, but the revision history evidences how the solution was built-up.

`README.md`: this file.

`interactive_runner.ipynb`: The interactive notebook is designed to make it easy to run the project.  Once opened in Colab, it will download the code from the respository and run the main function (see below).

`main.py`: This orchestrates the data pre-processing and the various learning tasks. It takes three arguments:
  * `tasks`: a list of tasks to run, e.g. "ACEFGHI"
  * `epochs`: the number of epochs for which to run each task (useful to set a low value for end-to-end testing)
  * `force:`: whether to force the data download and pre-processing steps to be repeated, even if not required

## Requirements

### Packages

For the most part, the standard environment on Kaggle and Google Colab was sufficient to run the project.  The two exceptions were:
  * `gdown`: this was required to download the modified dataset from the author's Google Drive account.
  * `docopt`: this was used to document the arguments to `main.py`, should someone want to run it from the command-line

Specifically, the following packages were used:

`dataclasses, datetime, gdown, google.colab, json, keras, matplotlib, numpy, os, pandas, pathlib, PIL, seaborn, shutil, sklearn, tensorflow, typing, zipfile`

### Infrastructure

The requirements of the project are:

* Diskspace: 5gb
* System memory: 30gb
* GPU memory: 16gb

It is not possible to run the project without access to a GPU.  Kaggle offers 30 hours of GPU-time per week.

## Running the Code

As noted above, the easiest way to run the code is to open `interactive_runner.ipynb` on Colab and run all the cells.


