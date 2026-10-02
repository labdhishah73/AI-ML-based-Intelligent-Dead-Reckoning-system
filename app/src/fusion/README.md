# Fusion module — Stages 1 & 3

Owned by the Sensor & Fusion Engineer role (see `dhruva_team_structure.html`
role 4). Implements orientation estimation and position fusion for the
prototype — deliberately simplified from the full architecture in
`DHRUVA_PROJECT_CONTEXT.md`, per the scope cuts documented in
`PROTOTYPE_PLAN.md` and `dhruva_prototype_showcase.html`'s component
inventory.

## Files

- `orientation.ts` — wraps the `ahrs` npm package's Madgwick filter. Not
  hand-derived quaternion math — a known-good published implementation.
- `kalmanFilter.ts` — `SimpleFusionFilter`: linear Kalman filter over
  `[x, y, v]` (local metres, local metres, speed m/s). **No bias states, no
  attitude in the state vector** — that's the v1 scope cut from the full
  design's UKF with `[position, velocity, attitude, IMU bias]`.
- `selfTest.ts` — not a Jest suite, a standalone script validating three
  behaviors before anyone wires this into the live app:
  1. Constant-velocity dead reckoning tracks correctly with no GPS.
  2. Position uncertainty (`getPositionUncertaintyMetres()` — what sizes the
     UI's confidence circle) grows during an outage and shrinks after a fix.
  3. A GPS fix disagreeing by 80m after a long outage pulls the estimate
     partway, not all the way, on the first correction — the
     re-acquisition-doesn't-snap behavior that's called out as the hardest
     engineering problem in the architecture.

  Run it with:
  ```
  npx tsc src/fusion/selfTest.ts src/fusion/kalmanFilter.ts --outDir .tmp_selftest --module commonjs --target es2019 --skipLibCheck --ignoreConfig
  node .tmp_selftest/selfTest.js
  rm -rf .tmp_selftest
  ```
  All three passed on 2026-09-05 (constant-velocity error <5m over 100m,
  uncertainty 4.47m→4.58m during a simulated 5s outage then down to 2.16m
  after one GPS fix, and an 80m disagreement pulled the estimate by 59.5m on
  the first correction rather than snapping the full 80m).

## The one thing this module exists to prove

`predict()` never reads GNSS. Only `updateGpsPosition()` does. This mirrors
the architecture's central claim (`DHRUVA_PROJECT_CONTEXT.md` section 4):
GNSS enters only as a correction, never the process model. In `App.tsx`,
the fusion tick calls `predict()` unconditionally on every timer tick,
regardless of whether a GPS fix has ever arrived — there's no branch on GPS
availability in the predict path. That's the whole point, not an
implementation detail.

## Day 2 update: the placeholder is gone

`speedModel.ts` now runs the real trained CNN-LSTM (`ml/speed_model.tflite`,
bundled at `src/assets/speed_model.tflite`) live on-device via
`react-native-fast-tflite` (Nitro-modules based — requires New Architecture,
already on in this project). `App.tsx` feeds it a rolling 100-sample window
of accel/gravity/gyro at 10Hz and uses its output as the fusion filter's
velocity observation, falling back to raw GPS speed only while the model is
still loading at app startup.

Confirmed the exported model's actual I/O shape before wiring the TS side
to it — input `[1, 100, 9]` float32, output `[1, 1]` float32, both
unquantized despite `Optimize.DEFAULT` (that flag only compressed weights,
given no representative dataset was provided) — so `speedModel.ts` reads
and writes plain `Float32Array`s with no dequantization step needed.

See `speedModel.ts`'s header comment for the exact input channel order and
the carried-forward assumption about gyro axis semantics (same unverified
Yaw/Pitch/Roll-vs-X/Y/Z question already flagged in `ml/schema.py` and
`ml/README.md` — not resolved here, just inherited consistently).

## What's honestly missing, not hidden

- No IMU bias states — see file header comment in `kalmanFilter.ts`.
- No adaptive Madgwick beta (see `orientation.ts` constructor doc) —
  `DHRUVA_PROJECT_CONTEXT.md` flags this as a known weakness during hard
  acceleration; not implemented here.
- No PCA-over-forward-acceleration yaw recovery — Madgwick's own heading
  output (gyro+accel+mag) is used directly instead.
- Non-holonomic constraint isn't implemented in this module at all yet —
  it's a post-check on the fused output, not a filter pseudo-measurement,
  and lives wherever the map-matching code ends up (Stage 4, not built yet).
