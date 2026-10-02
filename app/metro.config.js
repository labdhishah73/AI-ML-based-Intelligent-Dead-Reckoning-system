const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  resolver: {
    // Lets speed_model.tflite (see src/assets/) be require()'d as a bundled
    // asset by react-native-fast-tflite, the same way an image or font
    // would be. Without this, Metro doesn't know how to resolve .tflite
    // and require() throws at bundle time.
    assetExts: [...getDefaultConfig(__dirname).resolver.assetExts, 'tflite'],
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
