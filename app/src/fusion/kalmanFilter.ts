/**
 * Simplified position/velocity fusion filter — Stage 3 of the architecture
 * (see DHRUVA_PROJECT_CONTEXT.md "Stage 3 — GNSS+INS fusion").
 *
 * SCOPE CUT from the full design, documented in PROTOTYPE_PLAN.md and
 * dhruva_prototype_showcase.html's component inventory:
 *   - Full design: Unscented Kalman Filter, state = [position, velocity,
 *     attitude, IMU bias terms].
 *   - This prototype: linear Kalman filter, state = [x, y, v] (local
 *     meters, local meters, speed in m/s). No bias states, no attitude in
 *     the state vector (heading comes in externally each tick from
 *     OrientationFilter). Honest answer if asked: "bias estimation is
 *     architected — the state vector has the slots — we're validating the
 *     core fusion loop before adding it."
 *
 * Why this is still a real Kalman filter and not just smoothing: heading is
 * treated as a KNOWN external input at each timestep (from Madgwick), which
 * makes the [x, y, v] motion model linear in the state even though it
 * involves cos/sin — those become known coefficients per-step, not
 * functions of the state being estimated. No EKF linearization needed.
 *
 * GNSS enters only in updateGpsPosition() — the process model (predict())
 * never references it. This is deliberate: see DHRUVA_PROJECT_CONTEXT.md
 * section 4, "THE architectural point." An outage simply means
 * updateGpsPosition() stops being called; predict() keeps running exactly
 * as before.
 */

// 3x3 matrix helpers — deliberately inlined rather than pulling in a linear
// algebra dependency for a fixed 3x3 case.
type Mat3 = [[number, number, number], [number, number, number], [number, number, number]];
type Vec3 = [number, number, number];

function matMul(a: Mat3, b: Mat3): Mat3 {
  const r: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return r as Mat3;
}

function transpose(m: Mat3): Mat3 {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
  ];
}

function addMat(a: Mat3, b: Mat3): Mat3 {
  return a.map((row, i) => row.map((val, j) => val + b[i][j])) as Mat3;
}

export type FusionState = {
  x: number;      // local-frame east offset, metres
  y: number;      // local-frame north offset, metres
  v: number;      // speed, m/s
};

export type FusionConfig = {
  /** Process noise — how much we distrust the constant-velocity assumption
   * between updates. Higher = filter adapts faster to real speed changes
   * but trusts raw prediction less. */
  processNoisePosition: number;
  processNoiseVelocity: number;
};

const DEFAULT_CONFIG: FusionConfig = {
  processNoisePosition: 0.05,
  processNoiseVelocity: 0.5,
};

export class SimpleFusionFilter {
  private state: FusionState = { x: 0, y: 0, v: 0 };
  private P: Mat3 = [
    [10, 0, 0],
    [0, 10, 0],
    [0, 0, 5],
  ];
  private config: FusionConfig;

  constructor(config: Partial<FusionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  getState(): FusionState {
    return { ...this.state };
  }

  /** Filter's own uncertainty — this is literally what sizes the
   * confidence circle in the UI (dhruva_explained.html section 10). */
  getPositionUncertaintyMetres(): number {
    return Math.sqrt(this.P[0][0] + this.P[1][1]);
  }

  /**
   * Predict step — runs every tick regardless of GNSS availability.
   * headingRad comes from OrientationFilter; dtSec is elapsed time since
   * the previous predict() call.
   *
   * Guards against non-finite inputs on purpose: unlike OrientationFilter
   * (which can rebuild its own internal AHRS instance), this filter's state
   * has no equivalent "reset" — once x/y/v go NaN there's no recovering
   * without external re-initialization. A bad headingRad or dtSec (e.g.
   * from a momentarily corrupted upstream sensor read) is dropped as a
   * no-op tick rather than allowed to poison the position estimate forever.
   */
  predict(headingRad: number, dtSec: number): void {
    if (!Number.isFinite(headingRad) || !Number.isFinite(dtSec) || dtSec < 0) {
      return;
    }
    const { x, y, v } = this.state;
    const cosH = Math.cos(headingRad);
    const sinH = Math.sin(headingRad);

    this.state = {
      x: x + v * cosH * dtSec,
      y: y + v * sinH * dtSec,
      v,
    };

    // F is linear given headingRad as a known-per-step coefficient, not a
    // state variable — see file header for why this doesn't need EKF
    // linearization.
    const F: Mat3 = [
      [1, 0, cosH * dtSec],
      [0, 1, sinH * dtSec],
      [0, 0, 1],
    ];
    const Q: Mat3 = [
      [this.config.processNoisePosition * dtSec, 0, 0],
      [0, this.config.processNoisePosition * dtSec, 0],
      [0, 0, this.config.processNoiseVelocity * dtSec],
    ];

    this.P = addMat(matMul(matMul(F, this.P), transpose(F)), Q);
  }

  /**
   * Velocity observation from the trained speed model — enters as a
   * measurement, not written directly into state. See
   * DHRUVA_PROJECT_CONTEXT.md communication rules: "say velocity
   * observation, not speed prediction."
   */
  updateVelocityObservation(measuredV: number, measurementNoise: number): void {
    if (!Number.isFinite(measuredV) || !Number.isFinite(measurementNoise) || measurementNoise <= 0) {
      return;
    }
    // H = [0, 0, 1] — observes v only.
    const innovation = measuredV - this.state.v;
    const S = this.P[2][2] + measurementNoise;
    const K: Vec3 = [this.P[0][2] / S, this.P[1][2] / S, this.P[2][2] / S];

    this.state = {
      x: this.state.x + K[0] * innovation,
      y: this.state.y + K[1] * innovation,
      v: this.state.v + K[2] * innovation,
    };

    // P = (I - K H) P, specialized for H = [0,0,1]
    const newP: Mat3 = [
      [this.P[0][0] - K[0] * this.P[2][0], this.P[0][1] - K[0] * this.P[2][1], this.P[0][2] - K[0] * this.P[2][2]],
      [this.P[1][0] - K[1] * this.P[2][0], this.P[1][1] - K[1] * this.P[2][1], this.P[1][2] - K[1] * this.P[2][2]],
      [this.P[2][0] - K[2] * this.P[2][0], this.P[2][1] - K[2] * this.P[2][1], this.P[2][2] - K[2] * this.P[2][2]],
    ];
    this.P = newP;
  }

  /**
   * GNSS position correction — the ONLY place GNSS enters this filter.
   * measurementNoise should come from the deficit handler / GNSS quality
   * thresholds (satellite count, reported accuracy) — see
   * DHRUVA_PROJECT_CONTEXT.md Stage 5.
   */
  updateGpsPosition(measuredX: number, measuredY: number, measurementNoise: number): void {
    if (!Number.isFinite(measuredX) || !Number.isFinite(measuredY) || !Number.isFinite(measurementNoise) || measurementNoise <= 0) {
      return;
    }
    const innovX = measuredX - this.state.x;
    const innovY = measuredY - this.state.y;

    const Sx = this.P[0][0] + measurementNoise;
    const Sy = this.P[1][1] + measurementNoise;

    const Kx: Vec3 = [this.P[0][0] / Sx, this.P[1][0] / Sx, this.P[2][0] / Sx];
    const Ky: Vec3 = [this.P[0][1] / Sy, this.P[1][1] / Sy, this.P[2][1] / Sy];

    this.state = {
      x: this.state.x + Kx[0] * innovX + Ky[0] * innovY,
      y: this.state.y + Kx[1] * innovX + Ky[1] * innovY,
      v: this.state.v + Kx[2] * innovX + Ky[2] * innovY,
    };

    // Approximate joint update by shrinking both position variances toward
    // the measurement noise floor — a simplification appropriate for a
    // prototype filter with no cross-axis GPS correlation modeled.
    this.P[0][0] = (1 - Kx[0]) * this.P[0][0];
    this.P[1][1] = (1 - Ky[1]) * this.P[1][1];
  }
}
