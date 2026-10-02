/**
 * Manual Jest mock for react-native-sensors — same reason as the MapLibre
 * mock: the real package throws at import time when its native module
 * isn't linked into an actual running binary, which is always true inside
 * Jest. Provides just enough of the Observable-like API (`.subscribe()`
 * returning `{ unsubscribe }`) for App.tsx's smoke test to render without
 * ever actually receiving sensor data.
 */
const neverFiringSensor = {
  subscribe: () => ({ unsubscribe: () => {} }),
};

module.exports = {
  accelerometer: neverFiringSensor,
  gyroscope: neverFiringSensor,
  magnetometer: neverFiringSensor,
  gravity: neverFiringSensor,
  setUpdateIntervalForType: () => {},
  SensorTypes: {
    accelerometer: 'accelerometer',
    gyroscope: 'gyroscope',
    magnetometer: 'magnetometer',
    gravity: 'gravity',
  },
};
