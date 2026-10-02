"""
DHRUVA edge-deployable engine — standalone Python, no phone required.

Satisfies the problem statement's second required deliverable (see
DHRUVA_PROJECT_CONTEXT.md section 2, "Required deliverables" and
dhruva_prototype_showcase.html Scene 7): the same trained models running
standalone for external IMU data, sensor-agnostic by design. This is a
plain script, not a mobile app — it exists to prove the model isn't
phone-only, e.g. for a FOG-grade IMU feeding CSV/log data on a lab bench
or an embedded Linux target instead of an Android sensor stack.

Pipeline: raw IMU CSV -> windowed velocity inference (the same trained
CNN-LSTM model used in the app) -> naive heading-only dead reckoning using
that velocity -> position plot against true GPS, if the input file has it.

This intentionally reuses baseline_position_plot.py's naive integration
approach for position (no Kalman filter here) — the point of this script
is proving the *model* runs standalone, not re-implementing the full
fusion stack outside the app. The Kalman filter's real home is
app/src/fusion/kalmanFilter.ts, which is mobile-app-specific TypeScript,
not portable to "arbitrary external IMU on a lab bench" by design.

Usage:
    python edge_engine.py <path to S-*.csv> [--model speed_model.tflite] [--max-seconds N]

Runs equally well on a machine with no GPU, no Android tooling, no phone —
only numpy, pandas, matplotlib and the tflite runtime (bundled inside
tensorflow, or the separate lightweight `tflite-runtime` package on a
constrained target).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from schema import detect_s_structure, normalize_s, SPEED_MODEL_INPUT_COLUMNS

WINDOW_SIZE = 100
STRIDE = 10  # denser than training's stride=50 — for inference we want a
             # velocity estimate roughly every sample, not every half-window
EARTH_RADIUS_M = 6371000.0


def load_tflite_interpreter(model_path: Path):
    """Prefer the full tensorflow package (already a project dependency,
    see requirements.txt) so this script needs no extra install on a
    machine that already has the ml/ environment set up. Falls back to
    the standalone tflite-runtime package, which is what an actual
    resource-constrained edge target would carry instead of full TF."""
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=str(model_path))
    except ImportError as e:
        raise RuntimeError(
            "Neither tensorflow nor tflite_runtime is installed. "
            "On a real constrained edge target, install tflite-runtime only "
            "(pip install tflite-runtime) — no need for full TensorFlow there."
        ) from e


def infer_velocity_windows(interpreter, s_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run the trained model over sliding windows of the input file.
    Returns (center_row_indices, predicted_velocity_kmh) — one prediction
    per window, indexed at the window's center row so it can be aligned
    back against the original per-row timeline for plotting.
    """
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x = s_df[SPEED_MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
    n = len(x)

    centers = []
    preds = []
    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        window = x[start:start + WINDOW_SIZE][np.newaxis, :, :]  # (1, 100, 9)
        interpreter.set_tensor(input_details[0]["index"], window)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])
        preds.append(float(pred.flatten()[0]))
        centers.append(start + WINDOW_SIZE // 2)

    return np.array(centers), np.array(preds, dtype=np.float32)


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same equirectangular projection as ml/baseline_position_plot.py and
    app/App.tsx's latLonToLocalXY — kept consistent across all three so a
    comparison between them is meaningful."""
    lat0, lon0 = np.radians(lat[0]), np.radians(lon[0])
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    x = EARTH_RADIUS_M * (lon_r - lon0) * np.cos(lat0)
    y = EARTH_RADIUS_M * (lat_r - lat0)
    return x, y


def dead_reckon_from_model_velocity(s_df: pd.DataFrame, center_idx: np.ndarray, velocity_kmh: np.ndarray):
    """Integrate heading (from the phone's own orientation column — same
    simplification as the naive baseline, see baseline_position_plot.py)
    with the MODEL's velocity instead of double-integrated raw
    acceleration. This is the whole point of the speed model: no
    accumulating acceleration-bias error, because velocity comes from
    per-window recognition, not integration.
    """
    t_ms = s_df["time_since_start_ms"].to_numpy()[center_idx]
    yaw_deg = s_df["orient_1"].to_numpy()[center_idx]
    yaw_rad = np.radians(yaw_deg)
    v_ms = velocity_kmh / 3.6

    dt = np.diff(t_ms) / 1000.0
    dt = np.append(dt, dt[-1] if len(dt) else 0.0)

    vx = v_ms * np.cos(yaw_rad)
    vy = v_ms * np.sin(yaw_rad)
    x = np.cumsum(vx * dt)
    y = np.cumsum(vy * dt)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("s_csv_path", type=str, help="Path to an S-*.csv (or any IMU CSV matching the schema)")
    ap.add_argument("--model", type=str, default="speed_model.tflite")
    ap.add_argument("--max-seconds", type=float, default=None,
                     help="Restrict to the first N seconds of the first recording session (see the session-reset gotcha in ml/README.md)")
    args = ap.parse_args()

    s_path = Path(args.s_csv_path)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        print("Train it first: python train_speed_model.py --epochs 80", file=sys.stderr)
        sys.exit(1)

    s_raw = pd.read_csv(s_path, encoding="latin-1")
    structure = detect_s_structure(list(s_raw.columns))
    if structure is None:
        raise ValueError(f"Unrecognized column layout in {s_path} ({len(s_raw.columns)} cols)")
    s_df = normalize_s(s_raw, structure)

    # Same session-reset handling as baseline_position_plot.py — restrict
    # to the first contiguous recording session before doing anything else.
    session_breaks = np.where(np.diff(s_df["time_since_start_ms"].to_numpy()) < 0)[0]
    if len(session_breaks) > 0:
        first_break = int(session_breaks[0])
        print(f"Note: {len(session_breaks)} session reset(s) detected -- using only the first "
              f"session ({first_break + 1} of {len(s_df)} rows).")
        s_df = s_df.iloc[: first_break + 1].reset_index(drop=True)

    if args.max_seconds is not None:
        t0 = s_df["time_since_start_ms"].iloc[0]
        cutoff_ms = t0 + args.max_seconds * 1000.0
        mask = s_df["time_since_start_ms"] > cutoff_ms
        cutoff_idx = int(mask.idxmax()) if mask.any() else len(s_df)
        s_df = s_df.iloc[:cutoff_idx].reset_index(drop=True)

    if len(s_df) < WINDOW_SIZE:
        print(f"File too short after trimming ({len(s_df)} rows, need at least {WINDOW_SIZE}). "
              "Try without --max-seconds or pick a longer file.", file=sys.stderr)
        sys.exit(1)

    print(f"File: {s_path.name}  Structure: {structure.id}  Rows used: {len(s_df)}")
    print(f"Loading TFLite model: {model_path}")
    interpreter = load_tflite_interpreter(model_path)

    print("Running windowed inference (no phone, no Android runtime involved)...")
    centers, pred_v_kmh = infer_velocity_windows(interpreter, s_df)
    print(f"{len(pred_v_kmh)} velocity predictions produced "
          f"(mean {pred_v_kmh.mean():.1f} km/h, range {pred_v_kmh.min():.1f}-{pred_v_kmh.max():.1f} km/h)")

    model_x, model_y = dead_reckon_from_model_velocity(s_df, centers, pred_v_kmh)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    has_gps = s_df["gps_lat_deg"].notna().any() and (s_df["gps_lat_deg"] != 0).any()
    if has_gps:
        true_x, true_y = latlon_to_local_xy(s_df["gps_lat_deg"].to_numpy(), s_df["gps_lon_deg"].to_numpy())
        true_dist = np.sum(np.hypot(np.diff(true_x), np.diff(true_y)))
        final_error = np.hypot(model_x[-1] - true_x[-1], model_y[-1] - true_y[-1])
        drift_pct = 100.0 * final_error / true_dist if true_dist > 0 else float("nan")
        print(f"True path length: {true_dist:.1f} m")
        print(f"Final position error (model-driven DR vs GPS): {final_error:.1f} m")
        print(f"Drift: {drift_pct:.1f}% of distance travelled (target: <10%)")

        axes[0].plot(true_x, true_y, label="True GPS track", linewidth=2)
        axes[0].plot(model_x, model_y, label="Model-driven dead reckoning", linewidth=1.5, linestyle="--")
        axes[0].set_title(f"Position — drift {drift_pct:.1f}%")
    else:
        axes[0].plot(model_x, model_y, label="Model-driven dead reckoning", linewidth=1.5)
        axes[0].set_title("Position (no GPS reference in this file)")

    axes[0].set_xlabel("Local X (m)")
    axes[0].set_ylabel("Local Y (m)")
    axes[0].legend()
    axes[0].set_aspect("equal")

    axes[1].plot(pred_v_kmh, linewidth=1.2)
    axes[1].set_xlabel("window index")
    axes[1].set_ylabel("predicted velocity (km/h)")
    axes[1].set_title("Model velocity output over time")

    fig.suptitle(f"DHRUVA edge engine (standalone) — {s_path.name}")
    out_path = s_path.with_name(s_path.stem + ".edge_engine_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
