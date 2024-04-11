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
> This directory hosts the results.  There are subdirectories for the confusion matrices and learning rate curves.  The parent directory contains the full training history (`train_details.csv`), test scores (`experimental_results.csv`), the confusion matrix for the final emsemble model (`conf_mat_N.png`), and a learning rate curve comparison between the two best-of-breed models `learning_curves_best_of_breeds (bef fine-tuning).png`.

`/workings/`
> `assignment-workings.ipynb`: As noted in the 'Development Approach' section of the report, the author used this notebook to develop the solutions on Kaggle before carving the code into the module structure described here.  It isn't meant to be part of the assignment submission, per se, but the revision history evidences how the solution was built-up and has all the final test results recorded in the output.  It could be run in lieu of using `main.py` - the code is all equivalent.

`README.md`: this file.

`interactive_runner.ipynb`: The interactive notebook is designed to make it easy to run the project.  Once opened in Colab, it will download the code from the respository and run the main function (see below).  The author checked this in with the output of a more lightweight run of image_size=225 and epochs=10.

`main.py`: This orchestrates the data pre-processing and the various learning tasks. It takes three arguments (also documented in the script):
  * `tasks`: a list of tasks to run, e.g. "ACEFGHI" (or "all", or "none")
  * `image_size`: Image size on which to train the models (e.g. 255; useful to lower in lower memory environments)
  * `epochs`: the number of epochs for which to run each task (e.g. 75; useful to lower in lower memory environments)
  * `force:`: whether to force the data download and pre-processing steps to be repeated

## Requirements

### Packages

For the most part, the standard environment on Kaggle and Google Colab was sufficient to run the project.  The two exceptions were:
  * `gdown`: this was required to download the modified dataset from the author's Google Drive account
  * `docopt`: this was used to document the arguments to `main.py`, should someone want to run it from the command-line

Specifically, the following packages were used:

`dataclasses, datetime, docopt, gdown, google.colab, json, keras, matplotlib, numpy, os, pandas, pathlib, PIL, seaborn, shutil, sklearn, tensorflow, typing, zipfile`

### Infrastructure

The requirements of the project are:

* Diskspace: 5gb
* System memory: 30gb
* GPU memory: 16gb

It is not possible to run the project without access to a GPU.  Kaggle offers 30 hours of GPU-time per week.

## Running the Code

As noted above, the easiest way to run the code is to open `interactive_runner.ipynb` on Colab and run all the cells.

An alternative would be to run the `/workings/assignment-workings.ipynb` on either Colab or Kaggle.

## Ideas for Improving the Codebase

1. Add support for saving the state of the various model, so that re-running the analysis would be quicker.
2. Craft the utility methods into more classes/objects (e.g., perhaps a Task class would be elegant/useful).
3. Use tuning utilities, such as KerasTuner perhaps, to help automate some of the hyperparameter search.


