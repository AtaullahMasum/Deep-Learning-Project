"""
src/model.py

- build_transfer_model(...)  -> EfficientNetB0-based model with top classifier
- build_baseline_cnn(...)    -> small CNN from scratch (educational baseline)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_transfer_model(img_size=(224,224,3), dropout=0.3, dense_units=128, base_trainable=False):
    """
    Builds a transfer learning model with EfficientNetB0 as base.
    base_trainable: whether to set base model.trainable = True
    """
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=img_size, include_top=False, weights='imagenet'
    )
    base_model.trainable = base_trainable

    inputs = tf.keras.Input(shape=img_size)
    x = inputs
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name="effnetb0_cat_dog")
    return model

def build_baseline_cnn(input_shape=(128,128,3)):
    """
    Simple CNN baseline for experimentation.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs, name="baseline_cnn")
    return model
