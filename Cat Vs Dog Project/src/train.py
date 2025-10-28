"""
src/train.py

Run to train the model end-to-end.

Usage:
    python -m src.train

Adjust DATA_DIR and other config variables below.
"""

import os
import tensorflow as tf
from tensorflow.keras import callbacks
from pathlib import Path

from src.data import get_datasets
from src.model import build_transfer_model
from src.utils import plot_history

# -------------------------
# Config - edit as needed
# -------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root
# If you already have split dataset: set DATA_DIR to that location (with /train /val /test)
DATA_DIR = ROOT / "data"   # expected structure: data/train/* data/val/* data/test/*
# If you only have a source folder with classes, use create_split_from_source
SOURCE_DIR = ROOT / "raw"  # optional: raw/cats, raw/dogs
IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 12
EPOCHS_STAGE2 = 6
MODEL_DIR = ROOT / "saved_model"
BEST_MODEL_PATH = ROOT / "best_model.h5"
LOG_DIR = ROOT / "logs"
SEED = 42

if __name__ == "__main__":
    # optionally create split if raw exists and data doesn't
    if (not (DATA_DIR / "train").exists()) and SOURCE_DIR.exists():
        create_split_from_source(SOURCE_DIR, DATA_DIR, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED)

    # get datasets
    train_ds, val_ds, test_ds = get_datasets(str(DATA_DIR), img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED)

    # build model (transfer learning)
    model = build_transfer_model(img_size=IMG_SIZE + (3,), dropout=0.3, dense_units=128, base_trainable=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # callbacks
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = callbacks.ModelCheckpoint(str(BEST_MODEL_PATH), monitor='val_loss', save_best_only=True)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    tb = callbacks.TensorBoard(log_dir=str(LOG_DIR), histogram_freq=1)

    # Stage 1: train head
    print("=== Stage 1: training top layers (base frozen) ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        callbacks=[ckpt, es, reduce_lr, tb]
    )
    plot_history(history1, out_path=str(MODEL_DIR / "history_stage1.png"))

    # Stage 2: fine-tune some of the base model
    print("=== Stage 2: fine-tuning top of base model ===")
    base_model = model.layers[1]  # EfficientNet base is the second layer in our build_transfer_model
    # But to be robust, find by name:
    for layer in model.layers:
        if hasattr(layer, "name") and "efficientnet" in layer.name:
            base_model = layer
            break

    base_model.trainable = True
    # Freeze all but last N layers
    N = 30
    for layer in base_model.layers[:-N]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        callbacks=[ckpt, es, reduce_lr, tb]
    )
    plot_history(history2, out_path=str(MODEL_DIR / "history_stage2.png"))

    # Save final model (SavedModel format)
    final_save_path = MODEL_DIR / "cat_dog_effnet_saved"
    model.save(final_save_path, include_optimizer=False)
    print("Saved final model to", final_save_path)
