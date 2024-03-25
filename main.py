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
  DEFAULT_PARAMS = model_util.Params(100, 196, 2, True, 5, False)
  print(DEFAULT_PARAMS)

  ARTIFACTS_PATH = Path("artefacts").mkdir(parents=True, exist_ok=True)

  # Process Data
  cwd = os.getcwd()
  ds_train, ds_valid, ds_test, class_weights = data.data_preprocessing(Path(cwd), DEFAULT_PARAMS)
  print(f"Class Weights: {class_weights}")

  baseline_model = model_util.create_model(tf.keras.applications.ConvNeXtBase, "baseline_model", DEFAULT_PARAMS)
  df_train, df_test = model_util.run_task("convnextbase", baseline_model, ds_train, ds_valid, ds_test, DEFAULT_PARAMS)

  print(df_train)
  print(ds_test)
  
  # df_results = pd.DataFrame
  # ES = True
  # LR = False
  # IMAGE_SIZE = 255
  # BATCH_SIZE = 196

  # # ds_train, ds_valid = get_equal_split_aug_dataset_raw(dataset_path, 2)
  # (m, df_hist) = model_util.run_experiment_lr("use_pre_trained_model_convnext_tiny", 1, model_util.use_pre_trained_model_convnext_tiny, ds_train, ds_valid)
  # df_results = model_util.add_results(df_results, df_hist)
    
  # df_results.to_csv("results.csv")

  # m.evaluate(ds_test)


if __name__ == "__main__":
  cmd_args = core.handle_docopt_arguments(docopt(__doc__))
  main(**cmd_args)
