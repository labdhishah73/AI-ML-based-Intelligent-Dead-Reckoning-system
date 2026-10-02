# DHRUVA — ML data pipeline

Scaffolding for the ML track of the 3-day prototype (see `../PROTOTYPE_PLAN.md`).
Written against the confirmed IO-VNBD schema before the actual dataset files
were available locally — point `IOVNBD_ROOT` at wherever the dataset lands
and everything below should just run.

## Setup

**Python version matters.** TensorFlow has no build for Python 3.14 as of
this writing (too new). If `python3 --version` shows 3.14, install Python
3.12 alongside it and use a dedicated venv — don't fight this, it won't
resolve by waiting for a `pip install` retry:

```
winget install --id Python.Python.3.12 -e
py -3.12 -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; `.venv\Scripts\activate` in PowerShell/cmd
pip install -r requirements.txt
export IOVNBD_ROOT=/path/to/extracted/IO-VNBD     # PowerShell: $env:IOVNBD_ROOT = "..."
```

`IOVNBD_ROOT` must be the folder that directly contains `Synchronised V abd S
datasets` and `Unsynchronised V and S Dataset`. To get the actual dataset:

```
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/onyekpeu/IO-VNBD.git
cd IO-VNBD
git lfs pull --include="Synchronised V abd S datasets/Categorised IOVNB Dataset/**"
```
(The repo stores CSVs via Git LFS — a plain clone without `git lfs pull`
only gets placeholder pointer files. The `--include` scopes the pull to
what this project actually uses, a few GB instead of the full 30-40GB set.)

## Files

- `schema.py` — the 6 S-file column layouts and 1 V-file layout, reverse-
  engineered from `columns in dataset (1).docx` and
  `IO-VNBD_Column_Reference_Guide.docx`, normalized into two canonical
  schemas (`S_CANONICAL_COLUMNS`, `V_CANONICAL_COLUMNS`).
- `data_loader.py` — finds matched S/V file pairs, loads + normalizes them,
  and exposes a streaming window generator for training (never loads the
  full 30-40GB set into memory).
- `baseline_position_plot.py` — the "missing screening requirement" from
  the project context doc: naive double-integrated dead reckoning plotted
  against the true GPS track for one file. Run this first, on any one
  synchronised file pair, to get a number and a plot before any model exists.
- `train_speed_model.py` — trains the CNN-LSTM speed model and exports it to
  `speed_model.tflite`. See the "Speed model" section below for real timing
  and result numbers.
- `edge_engine.py` — the standalone edge-deployable engine deliverable: runs
  the same trained TFLite model on an IMU CSV with no phone, no Android
  runtime, no app involved at all. See its own section below.

## Edge-deployable engine (Sept 5)

```
python edge_engine.py "<path to an S-*.csv>" [--max-seconds N]
```

Confirmed working on real data: loads `speed_model.tflite`, runs windowed
inference (window 100 @ stride 10 — denser than training's stride 50,
since inference wants a velocity roughly every sample rather than every
half-window), then does heading-only dead reckoning using the *model's*
velocity instead of double-integrated acceleration. Produces a two-panel
plot: position vs true GPS, and the raw velocity output over time.

Real result on `S-M.csv`'s first session (44,226 rows after the session-
reset split): **47.8% drift**, vastly better than the naive baseline's
10,978% on the same file, and an honest, expected result — no map-matching
or GNSS fusion is applied here, so it's not meant to hit the <10% target on
its own. That's what stages 3-4 of the full architecture are for; this
script's job is proving the model runs standalone, not replacing the app's
fusion filter.

Runs on any machine with `tensorflow` installed (already a project
dependency) — falls back to the lighter `tflite-runtime` package if
`tensorflow` isn't available, which is the more realistic choice for an
actual resource-constrained edge target.

## Scope decision: synchronised tree only

`data_loader.find_synchronised_pairs()` only looks under `Synchronised V abd
S datasets`. Of Structure 1's 158 S files, 72 live there and are guaranteed
row-for-row aligned with their V counterpart. The `Unsynchronised V and S
Dataset` tree has no row-level time alignment between S and V files — using
it safely means interpolating both onto a common timeline by timestamp, which
is real work that isn't worth spending on a 3-day prototype. Revisit after
the prototype if more training data is needed.

## Scope decision: gyro/orientation axis convention

Structures 1/5/6 label the gyro and orientation triplets Yaw/Pitch/Roll;
structures 3/4 label the same column positions X/Y/Z and Azimuth/Pitch/Roll.
Whether these are the same physical axes is **not verified** — see the
warning in `DHRUVA_PROJECT_CONTEXT.md` section 6. The synchronised-tree
scope above happens to sidestep this entirely: all 72 synchronised-tree S
files are Structure 1. Don't mix in Structure 3/4 files without checking
this first.

## Data quality gotcha found while validating the pipeline (Sept 4)

At least one S file (`S-M.csv`) concatenates multiple recording sessions:
`TIME SINCE START (ms)` resets to ~10ms partway through instead of running
monotonically for the whole file (confirmed jump backward at row 44226).
`data_loader.split_sessions()` splits on any backward jump before windowing,
and `baseline_position_plot.py` restricts itself to the first session. If you
write any new script that slices by time value directly, don't use a boolean
filter (`df[df.time <= cutoff]`) — a later session's small timestamps can
alias into an earlier window and silently pull in the wrong rows. Slice by
row position after finding the session boundary instead.

## Confirmed working end-to-end (Sept 4)

Ran against the real dataset (synchronised tree, `M (Driver B)`): 72 pairs
discovered, schema normalization correct, baseline naive dead-reckoning
produces a real (bad, as expected) drift number and plot. CSVs are
Latin-1 encoded, not UTF-8 — both `data_loader.py` and
`baseline_position_plot.py` already read them with `encoding="latin-1"`.

## Speed model — trained, real numbers (Sept 5)

`train_speed_model.py` trains the CNN-LSTM speed & vibration model (Stage 2
— the load-bearing one). Run it with:

```
python train_speed_model.py --time-one-epoch    # measure real per-epoch time first
python train_speed_model.py --epochs 80          # early stopping will cut this short
```

**Measured on this machine (CPU only — no CUDA GPU, Intel Arc integrated
graphics, native Windows TensorFlow ≥2.11 has no GPU path at all without
WSL2/DirectML, neither of which was worth the setup risk under deadline
pressure):**

- Total dataset: 15,674 windows across 72 synchronised file pairs (window
  size 100 @ 10Hz = 10s, stride 50 = 50% overlap). Windowing all of it takes
  ~6 seconds — not the bottleneck.
- One epoch: **~2-20s** depending on whether data is loaded fresh from disk
  each epoch or materialized into memory once first (see below). The whole
  dataset is ~15.7k windows × (100, 9) float32 ≈ well under 100MB, so it's
  loaded into memory once via `load_windows()` rather than re-reading and
  re-parsing all 72 CSVs from a `tf.data.Dataset.from_generator` on every
  epoch — that mistake alone was costing ~15-20s/epoch in the first pass.
- **A full 80-epoch run (with early stopping, patience=8) took 2.8 minutes
  total, stopping at epoch 24.** This is not an overnight job — even a much
  more generous epoch budget finishes in well under an hour on this hardware.
- Best result so far: val_loss 338 (MSE), **val_MAE ≈ 13.5 km/h**. Real,
  honest, first-pass number — not tuned, not a final result. The
  predicted-vs-true plot (`speed_model_pred_vs_true.png`) shows the model
  tracks the general speed trend but is noisy and clips at the top of the
  validation set's speed range — an honest generalisation gap given the
  training set doesn't include everything the validation files contain.

**Two real bugs fixed getting here, both worth knowing if you touch this
file:**
1. TFLite conversion fails on a plain `LSTM` layer with
   `'tf.TensorListReserve' op requires element_shape to be static` — fixed
   by adding `unroll=True` to the LSTM layer. Since window size is fixed
   (100 steps), unrolling is free and avoids needing the SELECT_TF_OPS flex
   delegate on-device.
2. Without input normalization, gravity's near-constant ~9.8 m/s² magnitude
   dominates the loss surface next to gyro's much smaller rad/s values, and
   training plateaus almost immediately (observed: val_loss stopped
   improving after epoch 2 in the first, unnormalized pass). Fixed with a
   `tf.keras.layers.Normalization` layer using train-set-only mean/std,
   baked into the model itself so the exported TFLite model expects raw
   sensor units, not pre-normalized input.

TFLite export uses default (dynamic-range) quantization, not full int8 —
full-integer quantization with a representative dataset is a reasonable
next step but wasn't worth the added risk of hitting another LSTM
conversion issue this close to the deadline.

## Quickest path to a running baseline

```
python data_loader.py                       # lists discovered pairs, sanity check
python baseline_position_plot.py "<path to one S-*.csv from the printed list>"
```
