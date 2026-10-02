/**
 * Manual Jest mock for @maplibre/maplibre-react-native.
 *
 * The real package registers native TurboModules (MLRNCameraModule etc.)
 * that only exist inside an actual running app on a device/emulator —
 * Jest's test environment has no native binary at all, so
 * TurboModuleRegistry.getEnforcing() throws for any test that imports the
 * real package. This stub renders plain Views instead, which is all the
 * App.test.tsx smoke test ("renders correctly" without crashing) needs.
 */
const React = require('react');
const { View } = require('react-native');

const Map = ({ children, ...props }) => React.createElement(View, props, children);
const Camera = () => null;
const GeoJSONSource = ({ children }) => React.createElement(View, null, children);
const Layer = () => null;
const Marker = ({ children }) => React.createElement(View, null, children);

module.exports = { Map, Camera, GeoJSONSource, Layer, Marker };
