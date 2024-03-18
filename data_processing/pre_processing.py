import gdown
import keras
import pandas as pd
import shutil
import tensorflow as tf
import os
import zipfile

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory


# ultimately, need to pass these as params
IMAGE_SIZE = 255
BATCH_SIZE = 196


def data_preprocessing(path: Path):
    file = download_data() # pass data_path
    
    data_path = Path(path) / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(file, "r") as z:
        z.extractall(data_path)
        
    df_images = pd.read_csv((data_path / "train.csv"))
    
    X_train, X_test, y_train, y_test = train_test_split(df_images.image_id, df_images.label, test_size=0.2, random_state=12)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=12)
    
    train_path = create_ds_tree(X_train, y_train, data_path, "train")
    valid_path = create_ds_tree(X_valid, y_valid, data_path, "valid")
    test_path = create_ds_tree(X_test, y_test, data_path, "test")
    
    ds_train = create_dataset(train_path)
    ds_valid = create_dataset(valid_path)
    ds_test = create_dataset(test_path)
    
    # clean-up
    os.rmdir((data_path / "train_images"))
    os.remove(file)
                            
    return ds_train, ds_valid, ds_test


def download_data(path: Path) -> None:
    url = "https://drive.google.com/uc?id=1TJBf1HZxAMpowZ92BcgS5N_NPHE7LPOT"
    output = path / "data.zip"
    gdown.download(url, str(output), quiet=False)
    return output


def create_ds_tree(x, y, path, name):
    """
    
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


def create_dataset(path: Path):
    return image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
        batch_size=BATCH_SIZE,
    )   
