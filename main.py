# -*- coding: utf-8 -*-
"""
main assignment script

Usage:
  main [--download]

Options:
  --download  download data from source
"""
import datetime
import os
import pandas as pd
import tensorflow as tf

# handle different structure Kaggle (Notebook) vs. Colab (Modules)
# this wouldn't be kept in any "production" version.
try:
    from AMLS_II_assignment23_24.data_processing.pre_processing import data_preprocessing
    from AMLS_II_assignment23_24.model.util import Params, ResultCollector, create_model, run_task
except ModuleNotFoundError:
    pass

from docopt import docopt
from pathlib import Path
from tensorflow.keras.optimizers import Adam, AdamW


def main(download=False):

  tf.random.set_seed(67890)
  
  # Starting set of params
  params = Params(255, 196, 1, 0.005, True, 7, False, Adam)
  
  ARTEFACTS_PATH = Path("artefacts")
  ARTEFACTS_PATH.mkdir(parents=True, exist_ok=True)
  
  collector = ResultCollector(ARTEFACTS_PATH)
  collector.restore_results()
  
  # Process Data
  print("================")
  print("= Loading Data =")
  print("================")
  cwd = os.getcwd()
  ds_train, ds_valid, ds_test, class_weights = data_preprocessing(Path(cwd), params)
  print(f"Class Weights: {class_weights}")
  
  print("\n==== Task A: Explore Batch Size ====")
  #for bs in [64, 128, 192, 256]:
  for bs in [64, 128]:
      print(f"Batch Size: {bs}")
      ds_train = ds_train.rebatch(bs)
      ds_valid = ds_valid.rebatch(bs)
      model = create_model(tf.keras.applications.ConvNeXtTiny, params)
      run_task(f"A_{bs}", model, ds_train, ds_valid, ds_test, params, collector)


if __name__ == "__main__":
  cmd_args = core.handle_docopt_arguments(docopt(__doc__))
  main(**cmd_args)
