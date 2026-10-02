"""
Baseline dead-reckoning position plot on one IO-VNBD file pair.

This exists to satisfy the SIH26168 screening requirement flagged as the
highest-leverage open item in DHRUVA_PROJECT_CONTEXT.md section 7:
    "preliminary AI models and the results of the position plot inferenced
    from the subset of IO-VNBD dataset"

It is deliberately the dumbest possible dead reckoning: rotate raw
accelerometer into a world-ish frame using the phone's own orientation
columns, double-integrate to get a trajectory, and plot it against the true
GPS track from the same file. No trained model, no filtering — this is the
"look how badly naive integration drifts" baseline that everything else in
the project is justified against. Run this before the CNN-LSTM speed model
is ready; re-run with the trained model's velocity substituted in once it
exists, to show the improvement.

Usage:
    export IOVNBD_ROOT=<path to dataset root>
    python baseline_position_plot.py <path to one S-*.csv> [max_seconds]

    max_seconds is optional — truncates the run to the first N seconds of the
    file. Recommended for anything more than a few minutes long, since
    quadratic drift makes a full multi-hour file's error numerically huge
    and hard to read on a plot.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from schema import detect_s_structure, normalize_s

EARTH_RADIUS_M = 6371000.0


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection around the track's own start point — fine
    for a short drive, not for anything spanning a large area."""
    lat0, lon0 = np.radians(lat[0]), np.radians(lon[0])
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    x = EARTH_RADIUS_M * (lon_r - lon0) * np.cos(lat0)
    y = EARTH_RADIUS_M * (lat_r - lat0)
    return x, y


def naive_dead_reckoning(s_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Double-integrate horizontal acceleration in the device frame, coarsely
    rotated by the reported orientation yaw. This is intentionally naive —
    it is the failure case the rest of the architecture exists to fix.

    ACCELEROMETER readings include gravity; GRAVITY X/Y/Z is the phone's own
    estimate of that component. Subtracting it is the one correction kept
    even in this "naive" baseline — without it the horizontal channels carry
    a near-constant tilt-driven offset and the double integration blows up
    to physically meaningless numbers within seconds, which would make the
    baseline read as a bug rather than the expected quadratic drift failure.
    """
    dt = np.diff(s_df["time_since_start_ms"].to_numpy()) / 1000.0
    dt = np.append(dt, dt[-1] if len(dt) else 0.0)

    accel_x = s_df["accel_x"].to_numpy() - s_df["gravity_x"].to_numpy()
    accel_y = s_df["accel_y"].to_numpy() - s_df["gravity_y"].to_numpy()
    yaw_deg = s_df["orient_1"].to_numpy()  # yaw or azimuth depending on structure — see schema.py
    yaw_rad = np.radians(yaw_deg)

    # rotate device-frame accel into a world-ish frame using yaw only
    world_ax = accel_x * np.cos(yaw_rad) - accel_y * np.sin(yaw_rad)
    world_ay = accel_x * np.sin(yaw_rad) + accel_y * np.cos(yaw_rad)

    vx = np.cumsum(world_ax * dt)
    vy = np.cumsum(world_ay * dt)
    x = np.cumsum(vx * dt)
    y = np.cumsum(vy * dt)
    return x, y


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    s_path = Path(sys.argv[1])
    max_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else None

    s_raw = pd.read_csv(s_path, encoding="latin-1")
    structure = detect_s_structure(list(s_raw.columns))
    if structure is None:
        raise ValueError(f"Unrecognized column layout in {s_path} ({len(s_raw.columns)} cols)")
    s_df = normalize_s(s_raw, structure)

    # Some S files concatenate multiple recording sessions: TIME SINCE START
    # resets to ~0 partway through instead of running monotonically for the
    # whole file (confirmed on S-M.csv — the counter jumps backward at row
    # 44226). Always restrict to the first contiguous session before doing
    # anything else, or a later session's small timestamps can alias into an
    # earlier time window and silently corrupt any time-based slicing.
    session_breaks = np.where(np.diff(s_df["time_since_start_ms"].to_numpy()) < 0)[0]
    if len(session_breaks) > 0:
        first_break = int(session_breaks[0])
        print(f"Note: {len(session_breaks)} session reset(s) detected -- using only the first "
              f"session ({first_break + 1} of {len(s_df)} rows).")
        s_df = s_df.iloc[: first_break + 1].reset_index(drop=True)

    if max_seconds is not None:
        # Double-integrated dead reckoning drift grows with t^2 (see
        # DHRUVA_PROJECT_CONTEXT.md glossary). Over a multi-hour recording
        # the naive baseline's error becomes astronomically large and stops
        # being a legible number — truncate to a short, presentable segment
        # when you want a plot that reads clearly rather than a full-file
        # worst case. Positional slicing, not a value filter — a boolean
        # filter on the raw time value is exactly what breaks across a
        # session reset.
        t0 = s_df["time_since_start_ms"].iloc[0]
        cutoff_ms = t0 + max_seconds * 1000.0
        cutoff_idx = int((s_df["time_since_start_ms"] > cutoff_ms).idxmax()) if (s_df["time_since_start_ms"] > cutoff_ms).any() else len(s_df)
        s_df = s_df.iloc[:cutoff_idx].reset_index(drop=True)

    true_x, true_y = latlon_to_local_xy(s_df["gps_lat_deg"].to_numpy(), s_df["gps_lon_deg"].to_numpy())
    dr_x, dr_y = naive_dead_reckoning(s_df)

    true_dist = np.sum(np.hypot(np.diff(true_x), np.diff(true_y)))
    final_error = np.hypot(dr_x[-1] - true_x[-1], dr_y[-1] - true_y[-1])
    drift_pct = 100.0 * final_error / true_dist if true_dist > 0 else float("nan")

    print(f"File: {s_path.name}")
    print(f"Structure: {structure.id}  Rows: {len(s_df)}")
    print(f"True path length: {true_dist:.1f} m")
    print(f"Final position error (naive DR vs GPS): {final_error:.1f} m")
    print(f"Drift: {drift_pct:.1f}% of distance travelled  (target: <10%)")
    print("This number is expected to be bad -- that's the point of a baseline.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(true_x, true_y, label="True GPS track", linewidth=2)
    ax.plot(dr_x, dr_y, label="Naive dead reckoning (no filtering)", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_title(f"DHRUVA baseline — {s_path.name}\nDrift: {drift_pct:.1f}% (naive, unfiltered)")
    ax.legend()
    ax.set_aspect("equal")
    suffix = f".baseline_plot_{int(max_seconds)}s.png" if max_seconds is not None else ".baseline_plot_full.png"
    out_path = s_path.with_name(s_path.stem + suffix)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
