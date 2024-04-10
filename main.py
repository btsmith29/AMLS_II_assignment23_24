# -*- coding: utf-8 -*-
"""
main assignment script

Usage:
  main [--tasks=<tasks>] [--image_size=<image_size>] [--epochs=<epochs>] [--force_download]
  main (-h|--help)

Options:
    -h --help                   Show this help dialogue
    --tasks=<tasks>             Which tasks to run (e.g. "ABCEL", or "all", or "none") [default: "A"]
    ---image_size=<image_size>  Image size on which to train the models [default: 255]
    --epochs=<epochs>           Number of epochs to train for, as an upper bound [default: 75]
    --force_download            Force download of data even if it already exists locally
"""
import dataclasses
import datetime
import keras
import os
import pandas as pd
import tensorflow as tf

# handle different structure Kaggle (Notebook) vs. Colab (Modules)
# this wouldn't be kept in any "production" version.
try:
    from AMLS_II_assignment23_24.data_processing.pre_processing import (
      augment_dataset,
      cache_dataset,
      convert_dataset_to_float,
      data_preprocessing,
      over_sample_class,
    )
    from AMLS_II_assignment23_24.model.util import (
      Params, 
      ResultCollector, 
      create_model, 
      create_model_ensemble_avg, 
      create_simple_model,
      plot_task_comp_by_prefix,
      run_task,
    )
except ModuleNotFoundError:
    pass

from docopt import docopt
from pathlib import Path
from tensorflow.keras.optimizers import Adam, AdamW


def main(tasks:str="A", image_size:int=255, epochs:int=75, force_download=False):

  print(f"Tasks: {tasks}")

  tf.random.set_seed(67890)
  
  # Starting set of params
  params = Params(image_size, 196, epochs, 0.005, True, 7, False, Adam)
  
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
    plot_task_comp_by_prefix(collector, "A")

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
    plot_task_comp_by_prefix(collector, "B")

  # update based on results of Task B
  params.epsilon = 0.0075

  if _run_task(tasks, "C"):
    print("\n==== Task C: Baseline Model Comparison ====")
    for m in [tf.keras.applications.ConvNeXtTiny, tf.keras.applications.ConvNeXtBase,
              tf.keras.applications.EfficientNetB0, tf.keras.applications.EfficientNetV2B0]:
        print(f"Model: {m}")
        model = create_model(m, "C", params)
        run_task(f"C_{model.base_model.name}", model, ds_train_cached, ds_valid, ds_test, params, collector)
    plot_task_comp_by_prefix(collector, "C")

  # now used AdamW optimiser
  params.opt = AdamW

  # create some re-usable fine-tuning parameters
  ft_params = dataclasses.replace(params)
  ft_params.epochs = 1
  ft_params.epsilon = 1e-5
  ft_params.early_stopping_patience = 1  
  
  # oversample & augment dataset
  ds_train_aug_over = augment_dataset(over_sample_class(ds_train, 0, params.batch_size), 2)
  ds_train_aug = augment_dataset(ds_train, 2)
  
  if _run_task(tasks, "D"):
    print("\n==== Task D: Best-of-Breed Model ====")
    # initial training
    model_d = create_model(tf.keras.applications.EfficientNetV2B0, "D", params, batch_norm=True)
    run_task(f"D_init", model_d, ds_train_aug_over, ds_valid, ds_test, params, collector, class_weights)
    # fine-tune by allowing base model to be re-trained
    model_d.base_model.trainable = True
    run_task(f"D_tuned", model_d, ds_train_aug_over, ds_valid, ds_test, ft_params, collector, class_weights)

  print("\n==================")
  print("= Ablation Study =")
  print("==================")

  if _run_task(tasks, "E"):
    print("\n==== Task E: Remove Fine-Tuning ====")
    model = create_model(tf.keras.applications.EfficientNetV2B0, "E", params, batch_norm=True)
    run_task(f"E", model, ds_train_aug_over, ds_valid, ds_test, params, collector, class_weights)
    del model

  if _run_task(tasks, "F"):
    print("\n==== Task F: Remove over-sampling ====")
    model = create_model(tf.keras.applications.EfficientNetV2B0, "F", params, batch_norm=True)
    run_task(f"F", model, ds_train_aug, ds_valid, ds_test, params, collector, class_weights)
    del model

  if _run_task(tasks, "G"):
    print("\n==== Task G: Remove Data Augmentation ====")
    model = create_model(tf.keras.applications.EfficientNetV2B0, "G", params, batch_norm=True)
    run_task(f"G", model, ds_train_cached, ds_valid, ds_test, params, collector, class_weights)
    del model
    
  if _run_task(tasks, "H"):
    print("\n==== Task H: Remove Class Weights ====")
    model = create_model(tf.keras.applications.EfficientNetV2B0, "H", params, batch_norm=True)
    run_task(f"H", model, ds_train_cached, ds_valid, ds_test, params, collector)
    del model

  if _run_task(tasks, "I"):
    print("\n==== Task I: Remove Batch Norm ====")
    model = create_model(tf.keras.applications.EfficientNetV2B0, "I", params)
    run_task(f"I", model, ds_train_cached, ds_valid, ds_test, params, collector)
    del model

  # now regress to Adam
  params.opt = Adam

  if _run_task(tasks, "J"):
    print("\n==== Task J: Regress to the Adam Optimiser ====")
    model = create_model(tf.keras.applications.EfficientNetV2B0, "J", params)
    run_task(f"J", model, ds_train_cached, ds_valid, ds_test, params, collector)
    del model

  if _run_task(tasks, "K"):
    print("\n==== Task K: Remove a FC Layer ====")
    model = create_model(tf.keras.applications.EfficientNetV2B0, "K", params, 1)
    run_task(f"K", model, ds_train_cached, ds_valid, ds_test, params, collector)
    del model

  plot_task_comp(collector, ["D_init", "E", "F", "G", "H", "I", "J", "K"])

  if _run_task(tasks, "L"):
    print("\n==== Task L: Create a Custom Convnet ====")
    model = create_simple_model(params)
    simple_params = dataclasses.replace(params)
    simple_params.epochs = 5
    run_task(f"L", model,
             convert_dataset_to_float(ds_train),
             convert_dataset_to_float(ds_valid),
             convert_dataset_to_float(ds_test), simple_params, collector)

  # return to AdamW for best-of-breed
  params.opt = AdamW

  # create common input for later ensembles
  inputs = tf.keras.Input(shape=(params.image_size, params.image_size, 3))

  model_m = None
  if _run_task(tasks, "M"):
    print("\n==== Task M: New Best of Breed ====")
    model_m = create_model(tf.keras.applications.EfficientNetV2B0, "M", params, fc_layers=1, inputs=inputs)
    run_task(f"M_init", model_m, ds_train_aug, ds_valid, ds_test, params, collector, class_weights)
    # fine-tune  
    model_m.base_model.trainable = True
    run_task(f"M_tuned", model_m, ds_train_aug, ds_valid, ds_test, ft_params, collector, class_weights)
    model_m.base_model.trainable = False
    model_m.model.trainable = False

  if _run_task(tasks, "N"):
    print("\n==== Task N: Ensemble ====")
    convnext_base = create_model(tf.keras.applications.ConvNeXtBase, "N", params, fc_layers=1, inputs=inputs)
    run_task(f"N_train", convnext_base, ds_train, ds_valid, ds_test, params, collector, class_weights)
    convnext_base.base_model.trainable = False
    convnext_base.model.trainable = False
    model_n = create_model_ensemble_avg(params, inputs, [model_m, convnext_base])
    model_n.model.trainable = False
    run_task(f"N", model_n, ds_train, ds_valid, ds_test, params, collector, class_weights)

  print("\n=======================")
  print("= All Tasks Completed =")
  print("=======================")


def _run_task(selector: str, task: str):
    if (selector is None or selector == "none"):
        return False
    else:
        return (selector == "all") or (task in selector)


def _handle_docopt_arguments(args):
    """
    Remove double dash from docopt arguments, replace the single dash with underscore and remove --help

    Args:
        args (dict): dictionary of docopt arguments (ex: {'--ftp': True, '--upload': True, '--delete': False})

     Return:
         dictionary of docopt arguments
    """
    return {
        key.replace("--", "").replace("-", "_"): val
        for key, val in args.items()
        if key != "--help"
    }


if __name__ == "__main__":
  cmd_args = _handle_docopt_arguments(docopt(__doc__))
  main(**cmd_args)
