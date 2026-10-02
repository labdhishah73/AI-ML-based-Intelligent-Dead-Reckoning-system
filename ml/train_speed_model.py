"""
Trains the CNN-LSTM speed & vibration model — the one load-bearing trained
model in the architecture (see DHRUVA_PROJECT_CONTEXT.md Stage 2).

Input: windows of (accel_x, accel_y, accel_z, gravity_x, gravity_y, gravity_z,
gyro_1, gyro_2, gyro_3) — SPEED_MODEL_INPUT_COLUMNS from schema.py.
Label: mean V-side velocity over that window (SPEED_MODEL_LABEL_COLUMN).
Never fed GPS/S-file speed — see the data leakage rule in
DHRUVA_PROJECT_CONTEXT.md section 6.

Usage:
    export IOVNBD_ROOT=<path to dataset root>
    python train_speed_model.py [--epochs N] [--time-one-epoch]

--time-one-epoch runs a single epoch and exits, printing elapsed time so you
can extrapolate a real training time estimate instead of guessing.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import find_synchronised_pairs, streaming_dataset
from schema import SPEED_MODEL_INPUT_COLUMNS, SPEED_MODEL_LABEL_COLUMN

WINDOW_SIZE = 100   # 10 seconds at 10Hz
STRIDE = 50         # 50% overlap
N_CHANNELS = len(SPEED_MODEL_INPUT_COLUMNS)
BATCH_SIZE = 128
VAL_FRACTION = 0.15  # held out by file, not by window — avoids leaking a driving session across train/val


def load_windows(pairs) -> tuple[np.ndarray, np.ndarray]:
    """Materialize every window into memory once. The full dataset is ~15.7k
    windows of (100, 9) float32 — well under 100MB — so there's no reason to
    re-read and re-parse all 72 CSVs from disk on every training epoch, which
    is what a tf.data.Dataset.from_generator pipeline would otherwise do."""
    xs, ys = [], []
    for x, y in streaming_dataset(pairs, SPEED_MODEL_INPUT_COLUMNS, SPEED_MODEL_LABEL_COLUMN,
                                   window_size=WINDOW_SIZE, stride=STRIDE):
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.array(ys, dtype=np.float32)


def build_dataset(x, y, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(feature_mean, feature_std):
    # Raw channels have wildly different scales — gravity sits near a
    # constant ~9.8 m/s^2 while gyro rad/s values are often well under 1.
    # Without normalizing, gravity's large constant magnitude dominates the
    # loss surface and the model plateaus almost immediately (observed: val
    # loss stopped improving after epoch 2 in the unnormalized first pass).
    inputs = tf.keras.Input(shape=(WINDOW_SIZE, N_CHANNELS))
    norm = tf.keras.layers.Normalization(mean=feature_mean, variance=feature_std ** 2)
    x = norm(inputs)
    x = tf.keras.layers.Conv1D(32, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.LSTM(64, unroll=True)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--time-one-epoch", action="store_true",
                     help="Run exactly one epoch, print timing, and exit — for estimating total training time.")
    args = ap.parse_args()

    all_pairs = find_synchronised_pairs()
    n_val = max(1, int(len(all_pairs) * VAL_FRACTION))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]
    print(f"{len(train_pairs)} training files, {len(val_pairs)} validation files "
          f"(split by file, not by window, so no session leaks across the split)")

    print("Loading and windowing all files into memory (one-time cost, not per-epoch)...")
    t_load = time.time()
    train_x, train_y = load_windows(train_pairs)
    val_x, val_y = load_windows(val_pairs)
    print(f"Loaded {len(train_x)} train windows, {len(val_x)} val windows in {time.time()-t_load:.1f}s")

    # Normalization stats from the TRAINING set only — computing them from
    # val too would leak validation-set information into the model's inputs.
    feature_mean = train_x.reshape(-1, N_CHANNELS).mean(axis=0)
    feature_std = train_x.reshape(-1, N_CHANNELS).std(axis=0) + 1e-6

    train_ds = build_dataset(train_x, train_y, shuffle=True)
    val_ds = build_dataset(val_x, val_y, shuffle=False)

    model = build_model(feature_mean, feature_std)
    model.summary()

    epochs = 1 if args.time_one_epoch else args.epochs

    callbacks = []
    if not args.time_one_epoch:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint("speed_model_best.keras", monitor="val_loss", save_best_only=True),
        ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    elapsed = time.time() - t0

    if args.time_one_epoch:
        print(f"\nOne epoch took {elapsed:.1f}s ({elapsed/60:.1f} min) on this machine (CPU only, no CUDA GPU).")
        print(f"Extrapolated: 10 epochs ~ {elapsed*10/60:.1f} min, 30 epochs ~ {elapsed*30/60:.1f} min, "
              f"60 epochs ~ {elapsed*60/3600:.1f} hr.")
        return

    print(f"\nTotal training time: {elapsed/60:.1f} min for {epochs} epochs "
          f"({elapsed/epochs:.1f}s/epoch average).")

    model.save("speed_model.keras")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("DHRUVA speed model — training loss")
    ax.legend()
    fig.savefig("speed_model_loss_curve.png", dpi=150, bbox_inches="tight")
    print("Saved speed_model_loss_curve.png")

    # predicted-vs-true speed plot on the validation set — part of the
    # screening deliverable alongside the naive baseline plot
    plot_n = min(2000, len(val_x))
    pred_y = model.predict(val_x[:plot_n], batch_size=BATCH_SIZE).flatten()
    val_y_plot = val_y[:plot_n]

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(val_y_plot, label="true V-side velocity (km/h)", linewidth=1.2)
    ax2.plot(pred_y, label="predicted velocity (km/h)", linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("validation window index")
    ax2.set_ylabel("velocity (km/h)")
    ax2.set_title("DHRUVA speed model — predicted vs true (validation set)")
    ax2.legend()
    fig2.savefig("speed_model_pred_vs_true.png", dpi=150, bbox_inches="tight")
    print("Saved speed_model_pred_vs_true.png")

    # TFLite export, int8 post-training quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    Path("speed_model.tflite").write_bytes(tflite_model)
    print(f"Saved speed_model.tflite ({len(tflite_model)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
