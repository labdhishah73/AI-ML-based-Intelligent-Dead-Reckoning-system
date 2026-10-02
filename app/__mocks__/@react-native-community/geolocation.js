/**
 * Manual Jest mock for @react-native-community/geolocation — same reason
 * as the MapLibre and react-native-sensors mocks above: the real module
 * requires a linked native binary, which doesn't exist under Jest.
 */
module.exports = {
  watchPosition: () => 1,
  clearWatch: () => {},
  requestAuthorization: () => {},
  getCurrentPosition: (success) => {
    success({
      coords: { latitude: 0, longitude: 0, accuracy: 0, speed: null },
      timestamp: Date.now(),
    });
  },
};
