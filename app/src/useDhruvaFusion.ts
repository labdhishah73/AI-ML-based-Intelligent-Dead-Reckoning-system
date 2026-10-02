/**
 * The live sensor -> orientation -> speed model -> Kalman fusion pipeline,
 * as a single hook. Extracted out of App.tsx so the map view and the HUD
 * overlay can both read the same live state without each running its own
 * independent copy of the sensor subscriptions, GPS watch, and (critically)
 * its own separate TFLite model instance — that would be wasteful at best
 * and silently wrong at worst (two Kalman filters with diverging state).
 *
 * Call this once, in AppContent, and pass its return value down to
 * whatever needs to render it.
 */

import { useEffect, useRef, useState } from 'react';
import {
  accelerometer,
  gyroscope,
  magnetometer,
  gravity,
  setUpdateIntervalForType,
  SensorTypes,
} from 'react-native-sensors';
import { PermissionsAndroid, Platform } from 'react-native';
import Geolocation from '@react-native-community/geolocation';
import { OrientationFilter } from './fusion/orientation';
import { SimpleFusionFilter, FusionState } from './fusion/kalmanFilter';
import { loadSpeedModel, SpeedModel } from './fusion/speedModel';
import type { TensorflowModel } from 'react-native-fast-tflite';

// TARGET_UPDATE_HZ matches the problem statement's 10Hz mobile update rate
// (see DHRUVA_PROJECT_CONTEXT.md "Mandated benchmarks").
export const TARGET_UPDATE_HZ = 10;
export const UPDATE_INTERVAL_MS = 1000 / TARGET_UPDATE_HZ;

setUpdateIntervalForType(SensorTypes.accelerometer, UPDATE_INTERVAL_MS);
setUpdateIntervalForType(SensorTypes.gyroscope, UPDATE_INTERVAL_MS);
setUpdateIntervalForType(SensorTypes.magnetometer, UPDATE_INTERVAL_MS);
setUpdateIntervalForType(SensorTypes.gravity, UPDATE_INTERVAL_MS);

export type Vector3 = { x: number; y: number; z: number };
export type GpsFix = {
  lat: number;
  lon: number;
  accuracy: number;
  speedKmh: number | null;
};
export type LatLon = { lat: number; lon: number };

const EARTH_RADIUS_M = 6371000;

/** Equirectangular local-frame projection around a fixed origin — same
 * approach as ml/baseline_position_plot.py's latlon_to_local_xy, kept
 * consistent so field-test comparisons between the app and the offline
 * analysis scripts mean the same thing. Fine for a short demo route, not
 * for anything spanning a large area. */
function latLonToLocalXY(lat: number, lon: number, originLat: number, originLon: number) {
  const lat0 = (originLat * Math.PI) / 180;
  const lon0 = (originLon * Math.PI) / 180;
  const latR = (lat * Math.PI) / 180;
  const lonR = (lon * Math.PI) / 180;
  const x = EARTH_RADIUS_M * (lonR - lon0) * Math.cos(lat0);
  const y = EARTH_RADIUS_M * (latR - lat0);
  return { x, y };
}

/** Exact inverse of latLonToLocalXY — needed to put the fused position
 * (which the Kalman filter tracks in local metres, not lat/lon) back onto
 * the map, which only understands geographic coordinates. */
export function localXYToLatLon(x: number, y: number, originLat: number, originLon: number): LatLon {
  const lat0 = (originLat * Math.PI) / 180;
  const lat = originLat + (y / EARTH_RADIUS_M) * (180 / Math.PI);
  const lon = originLon + (x / (EARTH_RADIUS_M * Math.cos(lat0))) * (180 / Math.PI);
  return { lat, lon };
}

/** GPS accuracy (metres, 1-sigma-ish per Android's reporting) converted to
 * a measurement noise variance for the Kalman update — larger accuracy
 * numbers (worse fixes) get trusted less. Rule-based, not learned — see the
 * "GNSS trust weighting" row in dhruva_prototype_showcase.html's component
 * inventory for why this is a documented v1 simplification. */
function gpsAccuracyToNoiseVariance(accuracyMetres: number): number {
  return Math.max(accuracyMetres * accuracyMetres, 4);
}

async function requestLocationPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') {
    return true;
  }
  const granted = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
    {
      title: 'DHRUVA needs your location',
      message:
        'Used as the GNSS correction input to the fusion filter — never sent off the device.',
      buttonPositive: 'Allow',
    },
  );
  return granted === PermissionsAndroid.RESULTS.GRANTED;
}

export type DhruvaFusionState = {
  accel: Vector3 | null;
  gyro: Vector3 | null;
  mag: Vector3 | null;
  gps: GpsFix | null;
  gpsStatus: string;
  fused: FusionState | null;
  headingDeg: number | null;
  uncertaintyM: number | null;
  modelStatus: string;
  modelVelocityKmh: number | null;
  /** The fused position reprojected back to lat/lon, or null until the
   * first GPS fix establishes a local-frame origin. This is what actually
   * drives the map marker + confidence circle + camera. */
  fusedLatLon: LatLon | null;
};

export function useDhruvaFusion(): DhruvaFusionState {
  const [accel, setAccel] = useState<Vector3 | null>(null);
  const [gyro, setGyro] = useState<Vector3 | null>(null);
  const [mag, setMag] = useState<Vector3 | null>(null);
  const [gps, setGps] = useState<GpsFix | null>(null);
  const [gpsStatus, setGpsStatus] = useState('requesting permission…');
  const [fused, setFused] = useState<FusionState | null>(null);
  const [headingDeg, setHeadingDeg] = useState<number | null>(null);
  const [uncertaintyM, setUncertaintyM] = useState<number | null>(null);
  const [modelStatus, setModelStatus] = useState('loading…');
  const [modelVelocityKmh, setModelVelocityKmh] = useState<number | null>(null);
  const [fusedLatLon, setFusedLatLon] = useState<LatLon | null>(null);

  // Latest raw readings, kept in refs (not state) so the 10Hz fusion tick
  // always reads the freshest value without re-subscribing every render.
  const latestAccel = useRef<Vector3 | null>(null);
  const latestGyro = useRef<Vector3 | null>(null);
  const latestMag = useRef<Vector3 | null>(null);
  const latestGrav = useRef<Vector3 | null>(null);
  const latestGpsSpeedMs = useRef<number | null>(null);
  const originRef = useRef<LatLon | null>(null);
  const lastTickMsRef = useRef<number | null>(null);

  const orientationFilterRef = useRef<OrientationFilter | undefined>(undefined);
  const fusionFilterRef = useRef<SimpleFusionFilter | undefined>(undefined);
  const speedModelRef = useRef<SpeedModel | undefined>(undefined);
  if (!orientationFilterRef.current) {
    orientationFilterRef.current = new OrientationFilter(TARGET_UPDATE_HZ);
  }
  if (!fusionFilterRef.current) {
    fusionFilterRef.current = new SimpleFusionFilter();
  }

  // Throttled console logging so the JS thread isn't spammed at 10Hz*3
  // sensors — still enough to eyeball real values during Day 1 smoke tests.
  const lastLogRef = useRef(0);

  useEffect(() => {
    const accelSub = accelerometer.subscribe(v => {
      latestAccel.current = v;
      setAccel(v);
      maybeLog('accel', v);
    });
    const gyroSub = gyroscope.subscribe(v => {
      latestGyro.current = v;
      setGyro(v);
      maybeLog('gyro', v);
    });
    const magSub = magnetometer.subscribe(v => {
      latestMag.current = v;
      setMag(v);
      maybeLog('mag', v);
    });
    const gravSub = gravity.subscribe(v => {
      latestGrav.current = v;
      maybeLog('grav', v);
    });

    let cancelled = false;
    loadSpeedModel()
      .then((model: TensorflowModel) => {
        if (cancelled) {
          return;
        }
        speedModelRef.current = new SpeedModel(model);
        setModelStatus('loaded, filling window…');
      })
      .catch((err: Error) => {
        console.log('[speedModel] failed to load:', err.message);
        setModelStatus(`load failed: ${err.message}`);
      });

    function maybeLog(label: string, v: Vector3) {
      const now = Date.now();
      if (now - lastLogRef.current > 1000) {
        console.log(
          `[sensor] ${label} x=${v.x.toFixed(2)} y=${v.y.toFixed(2)} z=${v.z.toFixed(2)}`,
        );
        lastLogRef.current = now;
      }
    }

    // The fusion loop — runs at TARGET_UPDATE_HZ regardless of GPS
    // availability. This is the one piece of code in the whole app that
    // directly demonstrates "GNSS enters only as a correction, never the
    // process model": nothing here branches on whether `gps` is null.
    const tickInterval = setInterval(() => {
      const now = Date.now();
      const dt = lastTickMsRef.current ? (now - lastTickMsRef.current) / 1000 : UPDATE_INTERVAL_MS / 1000;
      lastTickMsRef.current = now;

      if (!latestAccel.current || !latestGyro.current) {
        return; // no sensor data yet
      }

      const euler = orientationFilterRef.current!.update(
        latestGyro.current,
        latestAccel.current,
        latestMag.current ?? undefined,
        dt,
      );
      const headingRad = (euler.headingDeg * Math.PI) / 180;
      setHeadingDeg(euler.headingDeg);

      fusionFilterRef.current!.predict(headingRad, dt);

      // Real velocity observation from the trained speed model. Per
      // DHRUVA_PROJECT_CONTEXT.md's communication rules ("say velocity
      // observation, not speed prediction"): this enters the filter as a
      // measurement with its own noise, not written directly into state.
      const model = speedModelRef.current;
      if (model && latestGrav.current) {
        model.pushSample({
          accelX: latestAccel.current.x, accelY: latestAccel.current.y, accelZ: latestAccel.current.z,
          gravX: latestGrav.current.x, gravY: latestGrav.current.y, gravZ: latestGrav.current.z,
          gyroX: latestGyro.current.x, gyroY: latestGyro.current.y, gyroZ: latestGyro.current.z,
        });
        const predictedMs = model.predictVelocityMs();
        if (predictedMs != null) {
          setModelStatus('running');
          setModelVelocityKmh(predictedMs * 3.6);
          // Measurement noise: fixed for now, not learned — the "adaptive
          // noise network" from the full architecture (Stage 3) is a
          // documented v1 simplification, same as the GNSS trust weighting
          // (see dhruva_prototype_showcase.html's component inventory).
          fusionFilterRef.current!.updateVelocityObservation(predictedMs, 4.0);
        }
      } else if (!model) {
        // Model still loading — fall back to raw GPS speed (or 0 with no
        // fix yet) so the fusion loop has *something* rather than stalling
        // entirely during the async load window at app startup.
        const fallbackSpeedMs = latestGpsSpeedMs.current ?? 0;
        fusionFilterRef.current!.updateVelocityObservation(fallbackSpeedMs, 4.0);
      }

      const state = fusionFilterRef.current!.getState();
      setFused(state);
      setUncertaintyM(fusionFilterRef.current!.getPositionUncertaintyMetres());

      if (originRef.current) {
        setFusedLatLon(localXYToLatLon(state.x, state.y, originRef.current.lat, originRef.current.lon));
      }
    }, UPDATE_INTERVAL_MS);

    let watchId: number | null = null;
    requestLocationPermission().then(granted => {
      if (!granted) {
        setGpsStatus('permission denied');
        return;
      }
      setGpsStatus('waiting for fix…');
      watchId = Geolocation.watchPosition(
        position => {
          const { latitude, longitude, accuracy, speed } = position.coords;
          latestGpsSpeedMs.current = speed;
          setGps({
            lat: latitude,
            lon: longitude,
            accuracy,
            speedKmh: speed != null ? speed * 3.6 : null,
          });
          setGpsStatus('live');
          console.log(
            `[sensor] gps lat=${latitude.toFixed(6)} lon=${longitude.toFixed(6)} acc=${accuracy?.toFixed(1)}m`,
          );

          if (!originRef.current) {
            originRef.current = { lat: latitude, lon: longitude };
            // First fix: the fused estimate starts at (0,0) in local
            // frame, i.e. exactly at the origin — seed the map position
            // immediately rather than waiting for the next tick.
            setFusedLatLon({ lat: latitude, lon: longitude });
          }
          const { x, y } = latLonToLocalXY(
            latitude, longitude,
            originRef.current.lat, originRef.current.lon,
          );
          fusionFilterRef.current!.updateGpsPosition(x, y, gpsAccuracyToNoiseVariance(accuracy));
        },
        error => setGpsStatus(`error: ${error.message}`),
        { enableHighAccuracy: true, distanceFilter: 0, interval: UPDATE_INTERVAL_MS },
      );
    });

    return () => {
      cancelled = true;
      accelSub.unsubscribe();
      gyroSub.unsubscribe();
      magSub.unsubscribe();
      gravSub.unsubscribe();
      clearInterval(tickInterval);
      if (watchId != null) {
        Geolocation.clearWatch(watchId);
      }
    };
  }, []);

  return {
    accel, gyro, mag, gps, gpsStatus, fused, headingDeg, uncertaintyM,
    modelStatus, modelVelocityKmh, fusedLatLon,
  };
}
