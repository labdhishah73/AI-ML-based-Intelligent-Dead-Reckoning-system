/**
 * Orientation estimation — Stage 1 of the architecture
 * (see DHRUVA_PROJECT_CONTEXT.md "Stage 1 — Alignment & calibration").
 *
 * Wraps the `ahrs` package's Madgwick implementation rather than
 * hand-deriving quaternion math — per team-structure guidance
 * ("use a known-good open-source implementation, don't write it from
 * scratch under this deadline").
 *
 * NOT included in this prototype (see dhruva_prototype_showcase.html
 * component inventory): the PCA-over-forward-acceleration step that
 * recovers vehicle yaw independent of magnetometer, and adaptive beta.
 * Both are documented next steps, not silently dropped.
 */

import AHRS from 'ahrs';

export type Vector3 = { x: number; y: number; z: number };

export type EulerAngles = {
  headingDeg: number;
  pitchDeg: number;
  rollDeg: number;
};

const RAD2DEG = 180 / Math.PI;

function isFiniteVector3(v: Vector3): boolean {
  return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
}

export class OrientationFilter {
  private ahrs: AHRS;
  private sampleRateHz: number;
  private beta: number;
  private lastGood: EulerAngles = { headingDeg: 0, pitchDeg: 0, rollDeg: 0 };

  /**
   * @param sampleRateHz expected update rate — used for AHRS's internal
   *   default dt when an explicit deltaTimeSec isn't passed to update().
   * @param beta Madgwick's single tuning dial (see glossary in
   *   dhruva_explained.html): higher trusts accel/mag more (faster
   *   correction, more noise), lower trusts gyro more (smoother, more
   *   drift). 0.4 is the library default; DHRUVA_PROJECT_CONTEXT.md flags
   *   this should become adaptive (lower during hard acceleration) —
   *   not implemented in this prototype.
   */
  constructor(sampleRateHz = 10, beta = 0.4) {
    this.sampleRateHz = sampleRateHz;
    this.beta = beta;
    this.ahrs = this.newAhrsInstance();
  }

  private newAhrsInstance(): AHRS {
    return new AHRS({
      sampleInterval: this.sampleRateHz,
      algorithm: 'Madgwick',
      beta: this.beta,
      doInitialisation: true,
    });
  }

  /**
   * Feed one synchronized sample. Accel/gravity units don't matter (the
   * filter normalizes internally) as long as they're consistent between
   * calls. Magnetometer is optional — omit it entirely on phones without
   * one (see DHRUVA_PROJECT_CONTEXT.md "Tier 3 fallback" for the
   * magnetometer-absent degradation path).
   *
   * Guards against a real failure seen on-device (a Nothing Phone, Sept 5
   * field test): a single non-finite sensor sample corrupts the AHRS
   * library's internal quaternion permanently — every update() call after
   * that returns NaN forever, since the quaternion state carries forward.
   * Two layers of defense: reject bad input before it reaches the filter,
   * and detect a NaN *output* (meaning the internal state was already
   * poisoned, from this call or an earlier one) and rebuild a fresh AHRS
   * instance rather than stay stuck returning NaN indefinitely.
   */
  update(gyro: Vector3, accel: Vector3, mag?: Vector3, deltaTimeSec?: number): EulerAngles {
    if (!isFiniteVector3(gyro) || !isFiniteVector3(accel) || (mag && !isFiniteVector3(mag))) {
      return this.lastGood;
    }
    if (deltaTimeSec !== undefined && !Number.isFinite(deltaTimeSec)) {
      deltaTimeSec = undefined;
    }

    if (mag) {
      this.ahrs.update(
        gyro.x, gyro.y, gyro.z,
        accel.x, accel.y, accel.z,
        mag.x, mag.y, mag.z,
        deltaTimeSec,
      );
    } else {
      this.ahrs.update(
        gyro.x, gyro.y, gyro.z,
        accel.x, accel.y, accel.z,
        undefined, undefined, undefined,
        deltaTimeSec,
      );
    }
    const { heading, pitch, roll } = this.ahrs.getEulerAngles();
    const result: EulerAngles = {
      headingDeg: heading * RAD2DEG,
      pitchDeg: pitch * RAD2DEG,
      rollDeg: roll * RAD2DEG,
    };

    if (!Number.isFinite(result.headingDeg) || !Number.isFinite(result.pitchDeg) || !Number.isFinite(result.rollDeg)) {
      this.ahrs = this.newAhrsInstance();
      return this.lastGood;
    }

    this.lastGood = result;
    return result;
  }

  getHeadingRad(): number {
    const { heading } = this.ahrs.getEulerAngles();
    return Number.isFinite(heading) ? heading : (this.lastGood.headingDeg * Math.PI) / 180;
  }
}
