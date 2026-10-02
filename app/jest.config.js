module.exports = {
  preset: '@react-native/jest-preset',
  // @maplibre/maplibre-react-native ships its native-component files as raw
  // TypeScript even in the "commonjs" build output (Metro's own codegen
  // transform handles them at app-build time — that's why the app itself
  // builds fine). Jest's default transformIgnorePatterns skips all of
  // node_modules, so without this override, requiring the package under
  // Jest fails with "Cannot use import statement outside a module."
  //
  // Extends (not replaces) the preset's own pattern — @react-native/jest-preset
  // already needs react-native/@react-native packages transformed too; a
  // naive override here clobbers that and breaks the preset's own setup file.
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?|@maplibre/maplibre-react-native|react-native-sensors)/)',
  ],
};
