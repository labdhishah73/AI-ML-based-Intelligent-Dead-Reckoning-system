"""
IO-VNBD column schema, reverse-engineered from:
  - columns in dataset (1).docx      (per-structure column dumps, 564 files, 6 structures)
  - IO-VNBD_Column_Reference_Guide.docx  (plain-language meaning + ML role per column)

Six S-file (smartphone) column layouts exist. Only one V-file (vehicle) layout exists.
This module normalizes all of them into two fixed canonical schemas so downstream
code never has to think about which structure a given CSV happens to be.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column names, in fixed order. This is what every S / V dataframe
# looks like after normalize_s() / normalize_v() runs, regardless of the raw
# structure it started as.
# ---------------------------------------------------------------------------

S_CANONICAL_COLUMNS = [
    "gps_lat_deg", "gps_lon_deg", "gps_alt_m", "gps_speed_kmh", "gps_accuracy_m",
    "gps_orientation_deg", "gps_satellites", "time_since_start_ms", "timestamp_raw",
    "accel_x", "accel_y", "accel_z",
    "gravity_x", "gravity_y", "gravity_z",
    "gyro_1", "gyro_2", "gyro_3",          # see `gyro_convention` below — DO NOT assume these equal yaw/pitch/roll across structures
    "mag_x", "mag_y", "mag_z",
    "orient_1", "orient_2", "orient_3",    # see `orient_convention` below
]

V_CANONICAL_COLUMNS = [
    "v_gps_satellites", "v_time_since_start_s", "v_lat_deg", "v_lon_deg",
    "v_velocity_kmh", "v_heading_deg", "v_height_km", "v_vertical_velocity_kmh",
    "v_sample_period_s", "v_steering_angle_deg",
    "v_wheel_speed_fl", "v_wheel_speed_fr", "v_wheel_speed_rl", "v_wheel_speed_rr",
    "v_yaw_rate_degs", "v_indicated_speed_kmh",
    "v_indicated_long_accel_g", "v_indicated_lat_accel_g",
    "v_handbrake", "v_gear_requested", "v_gear", "v_engine_rpm",
    "v_coolant_temp_c", "v_clutch", "v_brake_pressure_psi", "v_brake_position",
    "v_battery_voltage", "v_air_temp_c", "v_accel_pedal_position",
]

# ---------------------------------------------------------------------------
# IMPORTANT — the axis-naming gotcha called out in DHRUVA_PROJECT_CONTEXT.md:
# structures 1/5/6 label the gyroscope/orientation triplet Yaw/Pitch/Roll,
# structures 3/4 label the *same column position* X/Y/Z (gyro) and
# Azimuth/Pitch/Roll (orientation). These are NOT verified to be the same
# physical axes. For the prototype, gyro_convention/orient_convention are
# carried as metadata so training code can choose to restrict to one
# convention rather than silently mixing them.
# ---------------------------------------------------------------------------

@dataclass
class SStructure:
    id: int
    n_columns: int
    raw_columns: list[str]
    gyro_convention: str     # "ypr" or "xyz"
    orient_convention: str   # "yaw" or "azimuth"
    has_magnetometer: bool = True
    has_orientation: bool = True
    drop_trailing_unnamed: bool = False


STRUCTURE_1 = SStructure(
    id=1, n_columns=24, gyro_convention="ypr", orient_convention="yaw",
    raw_columns=[
        "GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)", "GPS ALTITUDE (m)",
        "GPS SPEED (Kmh)", "GPS ACCURACY (m)", "GPS ORIENTATION (°)",
        "GPS SATELLITES IN RANGE", "TIME SINCE START (ms)", "DATE (YYYY-MO-DD HH-MI-SS_SSS)",
        "ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)",
        "GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)",
        "GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)",
        "MAGNETIC FIELD X (μT)", "MAGNETIC FIELD Y (μT)", "MAGNETIC FIELD Z (μT)",
        "ORIENTATION (Yaw) (°)", "ORIENTATION (Pitch) (°)", "ORIENTATION (Roll ) (°)",
    ],
)

STRUCTURE_3 = SStructure(
    id=3, n_columns=24, gyro_convention="xyz", orient_convention="azimuth",
    raw_columns=[
        "GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)", "GPS ALTITUDE (m)",
        "GPS SPEED (Kmh)", "GPS ACCURACY (m)", "GPS ORIENTATION (°)",
        "GPS SATELLITES IN RANGE", "TIME SINCE START (ms)", "DATE (YYYY-MO-DD HH-MI-SS_SSS)",
        "ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)",
        "GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)",
        "GYROSCOPE X (rad/s)", "GYROSCOPE Y (rad/s)", "GYROSCOPE Z (rad/s)",
        "MAGNETIC FIELD X (μT)", "MAGNETIC FIELD Y (μT)", "MAGNETIC FIELD Z (μT)",
        "ORIENTATION (Azimuth) (°)", "ORIENTATION (Pitch) (°)", "ORIENTATION (Roll ) (°)",
    ],
)

# Structure 4 is Structure 3 with one cosmetic date-header typo — same layout.
STRUCTURE_4 = SStructure(
    id=4, n_columns=24, gyro_convention="xyz", orient_convention="azimuth",
    raw_columns=STRUCTURE_3.raw_columns,
)

# Structure 5 is Structure 1 plus one stray empty trailing column (pandas'
# "Unnamed: 24") — an export artifact, not real data.
STRUCTURE_5 = SStructure(
    id=5, n_columns=25, gyro_convention="ypr", orient_convention="yaw",
    raw_columns=STRUCTURE_1.raw_columns + ["Unnamed: 24"],
    drop_trailing_unnamed=True,
)

# Structure 6 drops magnetometer and orientation entirely (18 cols).
STRUCTURE_6 = SStructure(
    id=6, n_columns=18, gyro_convention="ypr", orient_convention="yaw",
    has_magnetometer=False, has_orientation=False,
    raw_columns=[
        "GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)", "GPS ALTITUDE (m)",
        "GPS SPEED (Kmh)", "GPS ACCURACY (m)", "GPS ORIENTATION (°)",
        "SATELLITES IN RANGE", "TIME SINCE START (ms)", "DATE (YYYY-MO-DD HH-MI-SS_SSS)",
        "ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)",
        "GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)",
        "GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)",
    ],
)

S_STRUCTURES = {s.id: s for s in [STRUCTURE_1, STRUCTURE_3, STRUCTURE_4, STRUCTURE_5, STRUCTURE_6]}

V_RAW_COLUMNS = [
    "No. GPS satellites", "Time since start of day", "Latitude", "Longitude",
    "Velocity", "Heading", "Height", "Vertical velocity", "Sample period",
    "Steering angle", "Wheel speed FL", "Wheel speed FR", "Wheel speed RL", "Wheel speed RR",
    "Yaw rate", "Indicated vehicle speed", "Indicated longitudinal acceleration",
    "Indicated lateral acceleration", "Handbrake", "Gear requested", "Gear",
    "Engine speed", "Coolant temperature", "Clutch", "Brake pressure", "Brake position",
    "Battery voltage", "Air temperature", "Accelerator pedal position",
]


def detect_s_structure(columns: list[str]) -> Optional[SStructure]:
    """Match a raw S-file header (as read by pandas) to a known structure by column count.
    Falls back to None if nothing matches — caller should inspect manually rather than guess.
    """
    n = len(columns)
    candidates = [s for s in S_STRUCTURES.values() if s.n_columns == n]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    # n_columns==24 is ambiguous between structure 1 and structure 3/4 — disambiguate
    # by checking whether the gyro columns say "Yaw" or "X".
    for c in candidates:
        if any("GYROSCOPE Yaw" in col for col in columns) and c.gyro_convention == "ypr":
            return c
        if any("GYROSCOPE X" in col for col in columns) and c.gyro_convention == "xyz":
            return c
    return candidates[0]


def normalize_s(df: pd.DataFrame, structure: SStructure) -> pd.DataFrame:
    """Rename an S dataframe's columns to the canonical S_CANONICAL_COLUMNS,
    padding missing sensors (structure 6) with NaN so every output dataframe
    has the same shape.
    """
    df = df.copy()
    if structure.drop_trailing_unnamed and len(df.columns) == 25:
        df = df.iloc[:, :24]

    target = list(S_CANONICAL_COLUMNS)
    if not structure.has_magnetometer or not structure.has_orientation:
        # structure 6: first 18 canonical slots map 1:1, the rest are absent
        present = target[:18]
        df.columns = present
        for missing_col in target[18:]:
            df[missing_col] = float("nan")
        df = df[target]
    else:
        df.columns = target

    df.attrs["structure_id"] = structure.id
    df.attrs["gyro_convention"] = structure.gyro_convention
    df.attrs["orient_convention"] = structure.orient_convention
    return df


def normalize_v(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = V_CANONICAL_COLUMNS
    return df


# Columns considered safe to feed the speed model as input (no ground-truth
# leakage — see DHRUVA_PROJECT_CONTEXT.md section 6, "data leakage risk").
SPEED_MODEL_INPUT_COLUMNS = [
    "accel_x", "accel_y", "accel_z",
    "gravity_x", "gravity_y", "gravity_z",
    "gyro_1", "gyro_2", "gyro_3",
]

# The label column when training against synchronised V data.
SPEED_MODEL_LABEL_COLUMN = "v_velocity_kmh"
