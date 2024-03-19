# -*- coding: utf-8 -*-
"""
main assignment script

Usage:
  main [--download]

Options:
  --download  download data from source
"""

import os

from data import pre_processing as data
from docopt import docopt
from model import util as model_util


def main(download=False):
  cwd = os.getcwd()
  ds_train, ds_valid, ds_test = data_preprocessing(cwd)
  
  df_results = pd.DataFrame
  ES = True
  LR = False
  IMAGE_SIZE = 255
  BATCH_SIZE = 196

  # ds_train, ds_valid = get_equal_split_aug_dataset_raw(dataset_path, 2)
  (m, df_hist) = run_experiment_lr("use_pre_trained_model_convnext_tiny", 1, use_pre_trained_model_convnext_tiny, ds_train, ds_valid)
  df_results = add_results(df_results, df_hist)
    
  df_results.to_csv("results.csv")

  m.evaluate(ds_test)


if __name__ == "__main__":
  cmd_args = core.handle_docopt_arguments(docopt(__doc__))
  main(**cmd_args)
