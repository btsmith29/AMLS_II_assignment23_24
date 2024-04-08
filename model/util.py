# -*- coding: utf-8 -*-
"""
Functions for creating and training models, used across the various tasks.
"""
import dataclasses
import datetime as dt
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import timedelta
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.data import Dataset
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D
from typing import NamedTuple, Tuple


@dataclasses.dataclass
class Params():
    """
    Job Parameters Struct
    """
    image_size: int
    batch_size: int
    epochs: int
    epsilon: float
    early_stopping: bool
    early_stopping_patience: int
    adjust_learning_rate: bool
    opt: type
        
        
class ResultCollector():
    """
    Utility class to collect up and output results from tasks.
    """
    
    TRAIN_DETAILS_FILE = "train_details.csv"
    TEST_SCORES_FILE = "test_scores.csv"
    
    def __init__(
        self,
        path: Path
    ):
        self.path = path
        self.train_details = pd.DataFrame
        self.test_scores = pd.DataFrame
        
    def get_path(self) -> Path:
        """ Returns the path to which the collectors stores results """
        return self.path

    def add_task_results(self, df_train, df_test) -> None:
        """ Add training details and test scores into the collector"""
        self._add_train_details(df_train)
        self._add_test_scores(df_test)
        
    def get_train_details(self) -> pd.DataFrame:
        """ Returns the training details collected """
        return self.train_details
               
    def get_test_scores(self) -> pd.DataFrame:
        """ Returns the test scores collected """
        return self.test_scores
    
    def restore_results(self, quietly = True) -> None:
        """ Loads results from the location of get_path()"""
        try:
            self.train_details = pd.read_csv(self.path / self.TRAIN_DETAILS_FILE)
            self.test_scores = pd.read_csv(self.path / self.TEST_SCORES_FILE)
        except FileNotFoundError:
            print("Unable to restore history - starting fresh")
            if not quietly:
                raise

    def _add_train_details(self, df: pd.DataFrame) -> None:
        if self.train_details.empty:
            self.train_details = df
        else:
            self.train_details = pd.concat([self.train_details, df])
        
        self._save(self.train_details, self.TRAIN_DETAILS_FILE)
        
    def _add_test_scores(self, df: pd.DataFrame) -> None:
        if self.test_scores.empty:
            self.test_scores = df
        else:
            self.test_scores = pd.concat([self.test_scores, df])
            
        self._save(self.test_scores, self.TEST_SCORES_FILE)
                
    def _save(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.path / name, index=False)
        
        
@dataclasses.dataclass
class ModelWrapper():
    """
    Utility class to hold the "outer" model, and the inner base model
    so that training can be fine-tuned if required.
    """    
    model: keras.Model
    base_model: keras.Model


class LayerNamer():
    """
    Utility class to provide syntatic sugar for naming layers.
    """
    def __init__(self, name: str):
        self.name = name
        self.id = 0
        
    def n(self, prefix: str = ""):
        self.id += 1
        return f"{prefix}-{self.name}-{self.id}"
        

# def create_model(base_model_fn: str, params: Params,
#                  fc_layers = 2, fc_neurons = 1024, batch_norm = False,
#                  inputs = None) -> ModelWrapper:
#     """
#     Create Keras application model, e.g.
#         tf.keras.applications.EfficientNetV2B0
#         tf.keras.applications.ConvNeXtBase
#     with a custom top.
#     """
#     if inputs is None:
#         inputs = keras.Input(shape=(params.image_size, params.image_size, 3))
#     # Base
#     base_model = base_model_fn(weights='imagenet', include_top=False)
#     base_model.trainable = False
#     # set training=F here per https://keras.io/guides/transfer_learning/
#     x = base_model(inputs, training=False)
#     # Head
#     x = GlobalAveragePooling2D()(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#     x = Flatten()(x)
    
#     l = 0
#     while (l < fc_layers):
#         x = Dense(fc_neurons, activation="relu")(x)
#         x = Dropout(0.5)(x)
#         l = l + 1
    
#     outputs = Dense(5, activation="softmax")(x)
#     model = keras.Model(inputs, outputs)

#     return ModelWrapper(model, base_model)


def create_model(base_model_fn: str, name: str, params: Params,
                 fc_layers = 2, fc_neurons = 1024, batch_norm = False,
                 inputs = None) -> ModelWrapper:
    """
    Create Keras application model, e.g.
        tf.keras.applications.EfficientNetV2B0
        tf.keras.applications.ConvNeXtBase
    with a custom top.
    """
    n = LayerNamer(name)
    if inputs is None:
        inputs = keras.Input(shape=(params.image_size, params.image_size, 3))
    # Base
    base_model = base_model_fn(weights='imagenet', include_top=False)
    base_model.trainable = False
    try:
        base_model.name = f"{base_model.name}-{name}"
    except AttributeError:
        base_model._name = f"{base_model.name}-{name}"

    # set training=F here per https://keras.io/guides/transfer_learning/
    x = base_model(inputs, training=False)
    # Head
    x = GlobalAveragePooling2D(name=n.n("pooling"))(x)
    if batch_norm:
        x = BatchNormalization(name=n.n("bn"))(x)
    x = Flatten(name=n.n("flatten"))(x)
    
    l = 0
    while (l < fc_layers):
        x = Dense(fc_neurons, activation="relu", name=n.n("dense"))(x)
        x = Dropout(0.5, name=n.n("dropout"))(x)
        l = l + 1
    
    outputs = Dense(5, activation="softmax", name=n.n("dense-activation"))(x)
    model = keras.Model(inputs, outputs)

    return ModelWrapper(model, base_model)


def create_simple_model(params: Params) -> Model:
    m = keras.Sequential([
        
        tf.keras.Input(shape=(params.image_size, params.image_size, 3)),
        
        # First Convolutional Block
        layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same'),
        layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding='same'),
        layers.MaxPool2D(),
        layers.Dropout(0.2),

        # Second Convolutional Block
        layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
        layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
        layers.MaxPool2D(),
        layers.Dropout(0.2),

        # Third Convolutional Block
        layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
        layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
        layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
        layers.MaxPool2D(),
        layers.Dropout(0.2),

        # Classifier Head
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(units=5, activation="softmax"),
    ])
    return ModelWrapper(m, None)


def create_model_ensemble_avg(params: Params, inputs, models: list[ModelWrapper]) -> ModelWrapper:
    """
    Creates an ensemble for the given models, averaging the output.
    """
    outputs = [m.model.outputs[0] for m in [model_a, model_b]]
    
    em_output = tf.keras.layers.Average()(outputs)
    em_model = tf.keras.Model(inputs=inputs, outputs=em_output)
    
    # just averaging ensembled models - doesn't need fitting.
    em_model.compile(
        optimizer=params.opt(epsilon=params.epsilon),
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )
    
    return ModelWrapper(em_model, None)


def run_task(task_id: str, model_wrapper: ModelWrapper,
             ds_train: Dataset, ds_valid: Dataset, ds_test: Dataset,
             params: Params, collector: ResultCollector, weights = None) -> None:
    """
    Main task running function.
    """
    print(f"Running Task: {task_id} with params {params}")
    model = model_wrapper.model
    # train
    start = dt.datetime.now()
    df_train = _train(task_id, model, ds_train, ds_valid, params)
    end = dt.datetime.now()
    # test
    test_result = model.evaluate(ds_test)
    df_test = _create_test_record(task_id, test_result, (end-start))
    # save
    collector.add_task_results(df_train, df_test)
    _save_confusion_matrix(collector.get_path(), ds_test, model, task_id)


def _train(task_id: str, model: Model,
             ds_train_: Dataset, ds_valid_: Dataset,
             params: Params, weights = None) -> pd.DataFrame:
    
    opt = params.opt
    print(f"Using: {opt}")

    model.compile(
        optimizer=params.opt(epsilon=params.epsilon),
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.0001,
        patience=params.early_stopping_patience,
        restore_best_weights=True,
        verbose = 1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor = 'val_loss', factor = 0.3, 
        patience = 3, min_delta = 0.0005, 
        mode = 'min', verbose = 1)
    
    cbs = []
    if params.early_stopping:
        print("Using EarlyStopping")
        cbs += [early_stopping]
    if params.adjust_learning_rate:
        print("Using ReduceLROnPlateau")
        cbs += [reduce_lr]

    history = model.fit(
        ds_train_,
        validation_data=ds_valid_,
        epochs=params.epochs,
        verbose=1,
        callbacks=cbs,
        class_weight=weights
    )
   
    df_hist = pd.DataFrame(history.history)
    df_hist["task_id"] = task_id
    df_hist["epoch"] = df_hist.index
   
    return df_hist


def _create_test_record(task_id: str, result: list[float], duration: timedelta):
    return pd.DataFrame({"task_id": [task_id], "test_loss" : [result[0]], "test_accuracy": [result[1]], "time_secs": [duration.seconds]})


def _save_confusion_matrix(path: Path, ds: Dataset, model: Model, task_id: str) -> None:
    filepath = f"conf_mat_{task_id}.png"
    filepath = path / filepath
    
    probabilities = model.predict(ds)
    predictions = np.argmax(probabilities, axis=1)

    one_hot_labels = np.concatenate([y for x, y in ds], axis=0)
    labels = [np.argmax(x) for x in one_hot_labels]
    
    result = confusion_matrix(labels, predictions, labels=[0,1,2,3,4], normalize='pred')
    disp = ConfusionMatrixDisplay(result, display_labels=[0,1,2,3,4])
    disp.plot()
    disp.ax_.set_title(task_id)
    
    print(f"Saving confusion matrix to {path}")
    disp.figure_.savefig(filepath, dpi=300)
