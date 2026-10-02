/**
 * On-device inference wrapper for the trained CNN-LSTM speed model — Stage 2
 * of the architecture (see DHRUVA_PROJECT_CONTEXT.md "Stage 2 — Speed &
 * vibration filter"). This is the "Integration Engineer" piece flagged in
 * PROTOTYPE_PLAN.md as the single highest-risk task: getting the trained
 * model (ml/speed_model.tflite, ml/train_speed_model.py) actually running
 * inference inside the live app, replacing the placeholder GPS-speed input
 * that App.tsx used through Day 1.
 *
 * Uses react-native-fast-tflite (Nitro-modules based, JSI, zero-copy
 * ArrayBuffers) rather than an older bridge-based TFLite binding — see
 * PROTOTYPE_PLAN.md's risk note on this exact integration step.
 *
 * ---
 * INPUT CHANNEL ORDER — must exactly match ml/schema.py's
 * SPEED_MODEL_INPUT_COLUMNS, since the model was trained on that column
 * order and has no way to know if channels are swapped:
 *   accel_x, accel_y, accel_z, gravity_x, gravity_y, gravity_z,
 *   gyro_1, gyro_2, gyro_3
 *
 * accel_x/y/z <-> react-native-sensors' `accelerometer` (Android
 * TYPE_ACCELEROMETER, m/s^2, includes gravity — matches the training
 * data's "ACCELEROMETER X/Y/Z" columns exactly).
 *
 * gravity_x/y/z <-> react-native-sensors' `gravity` (Android TYPE_GRAVITY,
 * m/s^2, a separate virtual sensor estimating just the gravity component —
 * confirmed the native Android module actually implements TYPE_GRAVITY,
 * not just accelerometer under another name).
 *
 * gyro_1/2/3 <-> react-native-sensors' `gyroscope` .x/.y/.z (Android
 * TYPE_GYROSCOPE, rad/s), fed in that raw axis order.
 *
 * IMPORTANT CARRIED-FORWARD ASSUMPTION: the synchronised training files
 * (Structure 1) label these three gyro columns "Yaw/Pitch/Roll" rather
 * than "X/Y/Z" — but ml/schema.py already flags that whether that's a
 * true world-frame yaw/pitch/roll decomposition or just an alternate name
 * for the same raw device-frame X/Y/Z axes is UNVERIFIED (see
 * DHRUVA_PROJECT_CONTEXT.md section 6 and ml/README.md's "gyro/orientation
 * axis convention" section). This code makes the same assumption schema.py
 * already makes implicitly: gyro_1/2/3 are the raw per-axis gyro readings
 * in whatever order the training data had them, and device gyro x/y/z are
 * fed in directly in that same column position order. If this assumption
 * is wrong, the fix is re-verifying axis semantics in the dataset, not
 * changing this file.
 */

import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';

export const SPEED_MODEL_WINDOW_SIZE = 100; // matches WINDOW_SIZE in ml/train_speed_model.py
const N_CHANNELS = 9;

export type SpeedModelSample = {
  accelX: number; accelY: number; accelZ: number;
  gravX: number; gravY: number; gravZ: number;
  gyroX: number; gyroY: number; gyroZ: number;
};

function isFiniteSample(s: SpeedModelSample): boolean {
  return (
    Number.isFinite(s.accelX) && Number.isFinite(s.accelY) && Number.isFinite(s.accelZ) &&
    Number.isFinite(s.gravX) && Number.isFinite(s.gravY) && Number.isFinite(s.gravZ) &&
    Number.isFinite(s.gyroX) && Number.isFinite(s.gyroY) && Number.isFinite(s.gyroZ)
  );
}

/**
 * Loads speed_model.tflite from the app bundle. Call once at app startup;
 * the returned model is reused across the whole session by SpeedModel.
 */
export async function loadSpeedModel(): Promise<TensorflowModel> {
  return loadTensorflowModel(require('../assets/speed_model.tflite') as number, []);
}

/**
 * Maintains a rolling window of the last SPEED_MODEL_WINDOW_SIZE sensor
 * samples and runs the trained model over it on demand. Unlike
 * ml/edge_engine.py's batch inference (which strides sparsely through a
 * whole recorded file), this runs continuously: every new sample slides
 * the window forward by one step, so predict() can be called every tick
 * for a fresh (though heavily overlapping) estimate — this is what an
 * actual live navigation UI needs, not what the offline analysis script
 * needs.
 */
export class SpeedModel {
  private model: TensorflowModel;
  private rows: number[][] = [];
  private lastGoodVelocityMs = 0;

  constructor(model: TensorflowModel) {
    this.model = model;
  }

  pushSample(sample: SpeedModelSample): void {
    if (!isFiniteSample(sample)) {
      return; // drop a corrupted sensor read rather than poison the window
    }
    this.rows.push([
      sample.accelX, sample.accelY, sample.accelZ,
      sample.gravX, sample.gravY, sample.gravZ,
      sample.gyroX, sample.gyroY, sample.gyroZ,
    ]);
    if (this.rows.length > SPEED_MODEL_WINDOW_SIZE) {
      this.rows.shift();
    }
  }

  isReady(): boolean {
    return this.rows.length >= SPEED_MODEL_WINDOW_SIZE;
  }

  /**
   * Runs inference over the current window and returns predicted velocity
   * in m/s (converted from the model's trained km/h output — see
   * SPEED_MODEL_LABEL_COLUMN = "v_velocity_kmh" in ml/schema.py), or null
   * if the window isn't full yet.
   *
   * Uses runSync rather than the async run() — this is called from a
   * 10Hz setInterval tick in App.tsx, and the model is small (194KB,
   * dynamic-range quantized), so a synchronous call avoids adding promise
   * scheduling overhead to a real-time loop. Falls back to the last known
   * good value (mirroring OrientationFilter's and SimpleFusionFilter's
   * NaN-guarding pattern) if the model ever produces a non-finite output —
   * defends against the same class of bug already found live on-device in
   * the orientation filter (see src/fusion/README.md).
   */
  predictVelocityMs(): number | null {
    if (!this.isReady()) {
      return null;
    }
    const flat = new Float32Array(SPEED_MODEL_WINDOW_SIZE * N_CHANNELS);
    for (let t = 0; t < SPEED_MODEL_WINDOW_SIZE; t++) {
      const row = this.rows[t];
      for (let c = 0; c < N_CHANNELS; c++) {
        flat[t * N_CHANNELS + c] = row[c];
      }
    }

    const outputs = this.model.runSync([flat.buffer]);
    const predictedKmh = new Float32Array(outputs[0])[0];

    if (!Number.isFinite(predictedKmh)) {
      return this.lastGoodVelocityMs;
    }
    const velocityMs = predictedKmh / 3.6;
    this.lastGoodVelocityMs = velocityMs;
    return velocityMs;
  }
}
