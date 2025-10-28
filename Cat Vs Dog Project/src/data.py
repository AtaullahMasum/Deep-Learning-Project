import os
import random
import shutil
import tensorflow as tf
from pathlib import Path
import keras

AUTOTUNE = tf.data.AUTOTUNE

def get_datasets(data_dir: str,
                 img_size =(224, 224), 
                 batch_size = 32, 
                 seed = 42, 
                 augment=True, 
                 cache= True):
    """
    Returns (train_ds, val_ds, test_ds) as tf.data.Dataset objects.
    Expects data_dir to have train/, val/, test/ subfolders with class subfolders.
    """
    data_dir = str(data_dir)
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
      os.path.join(data_dir, "train"),
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
      
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.08),
    ], name="data_augmentation")
    
    def preprocess(ds, training=False):
        ds = ds.map(lambda x,y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
        if training and augment:
            ds = ds.map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x,y: (tf.keras.applications.efficientnet.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
        if cache:
            ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = preprocess(train_ds, training=True)
    val_ds = preprocess(val_ds, training=False)
    test_ds = preprocess(test_ds, training=False)

    return train_ds, val_ds, test_ds
    
  
  
  