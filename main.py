# -*- coding: utf-8 -*-
"""
main assignment script

Usage:
  main [--download]

Options:
  --download  download data from source
"""

import os
import pandas as pd
import tensorflow as tf

from AMLS_II_assignment23_24.data_processing import pre_processing as data
from AMLS_II_assignment23_24.model import util as model_util
from docopt import docopt
from pathlib import Path


def main(download=False):

  tf.random.set_seed(67890)

  #DEFAULT_PARAMS = model_util.Params(255, 196, 50, True, 5, False)
  DEFAULT_PARAMS = model_util.Params(50, 196, 1, True, 5, False)
  print(DEFAULT_PARAMS)

  ARTIFACTS_PATH = Path("artefacts").mkdir(parents=True, exist_ok=True)

  # Process Data
  print("==== Loading Data ====")
  cwd = os.getcwd()
  ds_train, ds_valid, ds_test, class_weights = data.data_preprocessing(Path(cwd), DEFAULT_PARAMS)
  print(f"Class Weights: {class_weights}")

  print("==== Task A: Baseline Model ====")
  baseline_model = model_util.create_model(tf.keras.applications.ConvNeXtBase, "baseline_model", DEFAULT_PARAMS)
  df_train, df_test = model_util.run_task("task_a", baseline_model, ds_train, ds_valid, ds_test, DEFAULT_PARAMS)
  print(df_train)
  print(df_test)

  print("==== Task B: Baseline + Data Augmentation ====")
  ds_train_aug = data.augment_dataset(ds_train, 2)
  baseline_model2 = model_util.create_model(tf.keras.applications.ConvNeXtBase, "baseline_model", DEFAULT_PARAMS)
  df_train, df_test = model_util.run_task("task_b", baseline_model2, ds_train_aug, ds_valid, ds_test, DEFAULT_PARAMS)
  print(df_train)
  print(df_test)


if __name__ == "__main__":
  cmd_args = core.handle_docopt_arguments(docopt(__doc__))
  main(**cmd_args)
