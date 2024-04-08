# -*- coding: utf-8 -*-
"""
main assignment script

Usage:
  main [--download]

Options:
  --download  download data from source
"""
import dataclasses
import datetime
import os
import pandas as pd
import tensorflow as tf

# handle different structure Kaggle (Notebook) vs. Colab (Modules)
# this wouldn't be kept in any "production" version.
try:
    from AMLS_II_assignment23_24.data_processing.pre_processing import cache_dataset, data_preprocessing
    from AMLS_II_assignment23_24.model.util import Params, ResultCollector, create_model, run_task
except ModuleNotFoundError:
    pass

from docopt import docopt
from pathlib import Path
from tensorflow.keras.optimizers import Adam, AdamW


def main(tasks:str=None, epochs:int=1, download=False):

  tf.random.set_seed(67890)
  
  # Starting set of params
  params = Params(255, 196, epochs, 0.005, True, 7, False, Adam)
  
  ARTEFACTS_PATH = Path("artefacts")
  ARTEFACTS_PATH.mkdir(parents=True, exist_ok=True)
  
  collector = ResultCollector(ARTEFACTS_PATH)
  collector.restore_results()
  
  # Process Data
  print("================")
  print("= Loading Data =")
  print("================")
  cwd = Path(os.getcwd())
  ds_train, ds_valid, ds_test, class_weights = data_preprocessing(cwd, params)
  print(f"Class Weights: {class_weights}")

  if _run_task(tasks, "A"):
    print("\n==== Task A: Explore Batch Size ====")
    for bs in [64, 128, 196, 256]:
        print(f"Batch Size: {bs}")
        params.batch_size = bs
        ds_train, ds_valid, ds_test, class_weights = data_preprocessing(cwd, params)
        model = create_model(tf.keras.applications.EfficientNetB0, "A", params)
        run_task(f"A_{bs}", model, cache_dataset(ds_train), ds_valid, ds_test, params, collector)

  # update based on results of Task A, regenerating data cleans up batch-sizes
  params.batch_size = 192
  ds_train, ds_valid, ds_test, class_weights = data_preprocessing(cwd, params)
  ds_train_cached = cache_dataset(ds_train)

  if _run_task(tasks, "B"):
    print("\n==== Task B: Explore Epsilon ====")
    for e in [0.0025, 0.0050, 0.0075, 0.01]:
        print(f"Epsilon: {e}")
        p = dataclasses.replace(params)
        p.epsilon = e
        model = create_model(tf.keras.applications.EfficientNetB0, "B", p)
        run_task(f"B_{e}", model, ds_train_cached, ds_valid, ds_test, p, collector)

  # update based on results of Task B
  params.epsilon = 0.0075

  if _run_task(tasks, "C"):
    print("\n==== Task C: Baseline Model Comparison ====")
    for m in [tf.keras.applications.ConvNeXtTiny, tf.keras.applications.ConvNeXtBase,
              tf.keras.applications.EfficientNetB0, tf.keras.applications.EfficientNetV2B0]:
        print(f"Model: {m}")
        model = create_model(m, "C", params)
        run_task(f"C_{model.base_model.name}", model, ds_train_cached, ds_valid, ds_test, params, collector)


def _run_task(selector: str, task: str):
    if (selector is None or selector == "none"):
        return False
    else:
        return (selector == "all") or (task in selector)


if __name__ == "__main__":
  cmd_args = core.handle_docopt_arguments(docopt(__doc__))
  main(**cmd_args)
