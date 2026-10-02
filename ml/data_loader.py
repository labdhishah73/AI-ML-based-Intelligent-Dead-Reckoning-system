"""
Dataset discovery + loading for IO-VNBD.

Point this at the root folder of the extracted dataset (the folder that
directly contains "Synchronised V abd S datasets" and "Unsynchronised V and
S Dataset") via the IOVNBD_ROOT environment variable, or pass root= explicitly.

Scope decision for the prototype (see PROTOTYPE_PLAN.md): only the
"Synchronised V abd S datasets" tree is used for training. Its S files are
all Structure 1 (24 cols, Yaw/Pitch/Roll gyro convention) paired row-for-row
with Structure 2 V files. The "Unsynchronised" tree has no row-level time
alignment between S and V and would need timestamp interpolation to use
safely — out of scope until after the prototype.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from schema import detect_s_structure, normalize_s, normalize_v, S_STRUCTURES

SYNCHRONISED_SUBTREE = "Synchronised V abd S datasets"


def dataset_root() -> Path:
    root = os.environ.get("IOVNBD_ROOT")
    if not root:
        raise RuntimeError(
            "Set IOVNBD_ROOT to the folder containing 'Synchronised V abd S "
            "datasets' and 'Unsynchronised V and S Dataset' before running "
            "any data loading code."
        )
    p = Path(root)
    if not p.exists():
        raise RuntimeError(f"IOVNBD_ROOT={root!r} does not exist.")
    return p


@dataclass
class FilePair:
    s_path: Path
    v_path: Path
    group: str  # e.g. "S (Driver A)/S1" — useful for train/val split by drive, not by row


def find_synchronised_pairs(root: Optional[Path] = None) -> list[FilePair]:
    """Walk the synchronised tree and pair every S-*.csv with the V-*.csv
    that sits next to it in the same directory. Matching is case-insensitive
    on the filename stem after stripping the leading 'S-' / 'V-' prefix,
    because the raw dataset is inconsistent about case (e.g. S-Vta2.csv next
    to V-vta2.csv).
    """
    root = root or dataset_root()
    sync_root = root / SYNCHRONISED_SUBTREE
    if not sync_root.exists():
        raise RuntimeError(f"Expected {sync_root} to exist under IOVNBD_ROOT.")

    pairs: list[FilePair] = []
    for dirpath, _dirnames, filenames in os.walk(sync_root):
        s_files = {f[2:].lower(): f for f in filenames if f.lower().startswith("s-") and f.lower().endswith(".csv")}
        v_files = {f[2:].lower(): f for f in filenames if f.lower().startswith("v-") and f.lower().endswith(".csv")}
        for stem, s_name in s_files.items():
            v_name = v_files.get(stem)
            if v_name is None:
                continue
            d = Path(dirpath)
            group = str(d.relative_to(sync_root))
            pairs.append(FilePair(s_path=d / s_name, v_path=d / v_name, group=group))
    return pairs


def load_pair(pair: FilePair) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize one matched S/V file pair. Returns (s_df, v_df),
    both with canonical column names, same row count assumed (synchronised
    tree only — verify with len() before trusting row-for-row alignment on
    unfamiliar files).
    """
    s_raw = pd.read_csv(pair.s_path, encoding="latin-1")
    structure = detect_s_structure(list(s_raw.columns))
    if structure is None:
        raise ValueError(
            f"{pair.s_path}: unrecognized column layout ({len(s_raw.columns)} cols). "
            f"Known structures: {sorted(S_STRUCTURES.keys())}. Inspect manually before adding a new one."
        )
    s_df = normalize_s(s_raw, structure)

    v_raw = pd.read_csv(pair.v_path, encoding="latin-1")
    v_df = normalize_v(v_raw)

    if len(s_df) != len(v_df):
        raise ValueError(
            f"{pair.group}: S has {len(s_df)} rows, V has {len(v_df)} rows — "
            "not actually row-synchronised, don't zip them blindly."
        )
    return s_df, v_df


def split_sessions(s_df: pd.DataFrame, v_df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Some S files concatenate multiple recording sessions — TIME SINCE
    START resets to ~0 partway through instead of increasing monotonically
    for the whole file (confirmed on S-M.csv: the counter jumps backward at
    row 44226). A window that straddles a reset would show the sensors
    "teleporting" in time, which is exactly the kind of silent corruption
    the streaming loader must not hand to training. Split on every backward
    jump in TIME SINCE START and return one (s_segment, v_segment) pair per
    contiguous session, each still row-aligned since S and V share an index.
    """
    t = s_df["time_since_start_ms"].to_numpy()
    breaks = np.where(np.diff(t) < 0)[0]
    if len(breaks) == 0:
        return [(s_df, v_df)]

    bounds = [0] + [b + 1 for b in breaks] + [len(s_df)]
    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end - start > 1:
            segments.append((s_df.iloc[start:end].reset_index(drop=True),
                              v_df.iloc[start:end].reset_index(drop=True)))
    return segments


def window_generator(
    s_df: pd.DataFrame,
    v_df: pd.DataFrame,
    input_columns: list[str],
    label_column: str,
    window_size: int = 100,
    stride: int = 50,
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (window, label) pairs for speed-model training. `window` has
    shape (window_size, len(input_columns)); `label` is the mean V-side
    value over that same window — never an S-side/GPS column, to avoid the
    data-leakage failure mode called out in DHRUVA_PROJECT_CONTEXT.md.
    """
    x = s_df[input_columns].to_numpy(dtype=np.float32)
    y = v_df[label_column].to_numpy(dtype=np.float32)
    n = len(x)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        yield x[start:end], float(np.mean(y[start:end]))


def streaming_dataset(pairs: list[FilePair], input_columns: list[str], label_column: str,
                       window_size: int = 100, stride: int = 50) -> Iterator[tuple[np.ndarray, float]]:
    """Generator over an entire list of file pairs, loading one file at a
    time (not all 30-40GB into memory at once) — feed this straight into
    tf.data.Dataset.from_generator for training.
    """
    for pair in pairs:
        try:
            s_df, v_df = load_pair(pair)
        except ValueError as e:
            print(f"[skip] {e}")
            continue
        for s_seg, v_seg in split_sessions(s_df, v_df):
            yield from window_generator(s_seg, v_seg, input_columns, label_column, window_size, stride)


if __name__ == "__main__":
    pairs = find_synchronised_pairs()
    print(f"Found {len(pairs)} synchronised S/V file pairs.")
    for p in pairs[:10]:
        print(" ", p.group, "->", p.s_path.name, p.v_path.name)
