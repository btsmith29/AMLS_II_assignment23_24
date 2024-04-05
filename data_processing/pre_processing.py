import gdown
import keras
import pandas as pd
import shutil
import tensorflow as tf
import os
import zipfile

# handle different structure Kaggle (Notebook) vs. Colab (Modules)
# this wouldn't be kept in any "production" version.
try:
    from AMLS_II_assignment23_24.model.util import Params
except ModuleNotFoundError:
    pass

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from typing import Tuple


def data_preprocessing(path: Path, params: Params, force=False) -> Tuple[Dataset, Dataset, Dataset, dict]:
    """
    Main data preprocessing function - extracts the data to the given path.
    Returns a tuple of Training, Validation, Test datasets, along with class weights.
    """
    file = _download_data(path, force)
    
    data_path = path / "data"
    if force:
        shutil.rmtree(data_path)
        
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
       
        with zipfile.ZipFile(file, "r") as z:
            z.extractall(data_path)
        
    df_images = pd.read_csv((data_path / "train.csv"))
    
    X_train, X_test, y_train, y_test = train_test_split(df_images.image_id, df_images.label, test_size=0.2, random_state=12)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=12)
    
    train_path = _create_ds_tree(X_train, y_train, data_path, "train")
    valid_path = _create_ds_tree(X_valid, y_valid, data_path, "valid")
    test_path = _create_ds_tree(X_test, y_test, data_path, "test")
    
    ds_train = _create_dataset(train_path, params.image_size, params.batch_size)
    ds_valid = _create_dataset(valid_path, params.image_size, params.batch_size)
    ds_test = _create_dataset(test_path, params.image_size, params.batch_size, False)

    return ds_train, ds_valid, ds_test, _extract_class_weights(df_images)


def _download_data(path: Path, force=False) -> Path:
    """
    Downloads the data from the author's Google Drive account.
    """
    url = "https://drive.google.com/uc?id=1TJBf1HZxAMpowZ92BcgS5N_NPHE7LPOT"
    output = path / "data.zip"
    if not Path(output).exists() or force:
        gdown.download(url, str(output), quiet=False)
    return output


def _create_ds_tree(x, y, path: Path, name: str) -> Path:
    """
    Creates the directory structure for the given dataset.
    """
    ds_path = path / name
    if not ds_path.exists():
        ds_path.mkdir(parents=True, exist_ok=True)

        for lab in y.unique():
            (ds_path / str(lab)).mkdir(exist_ok=True)

        source_path = path / "train_images"
        
        for img, lab in zip(x, y):
            src = source_path / img
            dest = ds_path / str(lab) / img
            shutil.move(src, dest)
        
    return ds_path


def _create_dataset(path: Path, img_size: int, batch_size: int, shuffle = True) -> Dataset:
    """
    Builds up the Dataset object from the given path.
    """
    return image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        image_size=[img_size, img_size],
        batch_size=batch_size,
        seed=12345,
        shuffle=shuffle,
        crop_to_aspect_ratio=True
    )


def _extract_class_weights(df_data: pd.DataFrame) -> dict:
    """
    Uses the descriptive DataFrame to calculate the class weights
    from the distribution of the labels.
    """
    classes = df_data.label.unique()
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=df_data.label)

    return dict(zip(classes, class_weights))


def convert_dataset_to_float(ds: Dataset) -> Dataset:
    """
    Some models require the input to be coverted to float tensors and
    normalised into a 0-1 range.
    """
    def convert_to_float(image, label):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image / 255.0
        return image, label

    return ds.map(convert_to_float)
    

def cache_dataset(ds: Dataset) -> Dataset:
    """
    Dataset caching/pre-fetch utility.
    """
    return (
        ds
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )


def augment_dataset(ds: Dataset, num_repeats: int = 1) -> Dataset:
    """
    Augment the given dataset by flipping left/right, up/down, and
    adjusting the brightness.
    """
    def augment(image, label):
        seed = 12345
        image = tf.image.random_flip_left_right(image, seed)
        image = tf.image.random_flip_up_down(image, seed)
        image = tf.image.random_brightness(image, 0.2, seed)
        return image, label

    return (
        ds
        .repeat(num_repeats)
        .map(augment)
    )

def over_sample_class(ds: Dataset, class_label: int, batch_size: int, num_repeats: int = 1) -> Dataset:
    """
    Over-samples the given class label by the number of repeats given.  Re-batch to the given size.
    Returns a combined, reshuffled dataset.
    """
    # filter dataset to just the class_label
    ds_filt = ds.unbatch().filter(lambda x, label: tf.equal(tf.argmax(label, axis=0), class_label))
    ds_filt = ds.repeat(num_repeats)
    # combined with original dataset, re-shuffle, and re-batch
    ds_over = tf.data.Dataset.concatenate(ds.unbatch(), ds_filt)
    ds_over = ds_over.shuffle(100000)
    ds_over = ds_over.batch(batch_size)
    return ds_over
