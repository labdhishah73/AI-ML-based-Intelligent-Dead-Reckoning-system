/**
 * DHRUVA — prototype app shell.
 *
 * Day 1: map renders, raw sensor streams flow into JS, orientation + fusion
 * filters run live with GPS speed standing in for the trained model's
 * velocity observation.
 *
 * Day 2: the placeholder is gone. The trained CNN-LSTM speed model
 * (ml/speed_model.tflite) runs live on-device via src/fusion/speedModel.ts,
 * feeding the fusion filter's velocity observation directly.
 *
 * Day 2b (this file, now): the fused position is wired onto the map as an
 * actual marker, with a real geographic confidence circle (a GeoJSON
 * polygon sized in true metres from the filter's own covariance, not a
 * fixed-size screen icon) — see dhruva_explained.html section 10 on why
 * this circle is "more honest than Google Maps' confident dot." The camera
 * follows the marker. Offline MBTiles replacing the live-network demo
 * style is still the Systems/Infra Engineer's open item, not this file's.
 *
 * @format
 */

import React from 'react';
import { Platform, StatusBar, StyleSheet, Text, View } from 'react-native';
import {
  SafeAreaProvider,
  useSafeAreaInsets,
} from 'react-native-safe-area-context';
import {
  Map as MapLibreMap,
  Camera as MapLibreCamera,
  GeoJSONSource,
  Layer,
  Marker,
} from '@maplibre/maplibre-react-native';
import {
  useDhruvaFusion,
  localXYToLatLon,
  Vector3,
  GpsFix,
  LatLon,
} from './src/useDhruvaFusion';

// PLACEHOLDER map style — MapLibre's public demo tiles, used only so the
// map renders during scaffolding. This makes a live network call, which
// contradicts the "zero network calls during navigation" thesis in
// DHRUVA_PROJECT_CONTEXT.md — the Systems/Infra Engineer's offline MBTiles
// extract replaces this before the real demo. Do not present this build
// as the offline claim.
const PLACEHOLDER_MAP_STYLE =
  'https://demotiles.maplibre.org/style.json';

const DEFAULT_ZOOM = 17; // close enough that a single-digit-metre confidence circle is actually visible

/** Builds a real geographic circle (a GeoJSON Polygon, not a fixed-pixel
 * icon) of the given radius in metres around a centre point, by offsetting
 * numPoints points around the centre using the same small-area
 * equirectangular approximation as localXYToLatLon/latLonToLocalXY. Good
 * enough for confidence radii of tens to low-hundreds of metres; not a
 * geodesically exact circle for anything larger. */
function makeConfidenceCirclePolygon(
  center: LatLon,
  radiusMetres: number,
  numPoints = 48,
): GeoJSON.Feature<GeoJSON.Polygon> {
  const coords: [number, number][] = [];
  for (let i = 0; i <= numPoints; i++) {
    const angle = (i / numPoints) * 2 * Math.PI;
    const dx = radiusMetres * Math.cos(angle);
    const dy = radiusMetres * Math.sin(angle);
    const { lat, lon } = localXYToLatLon(dx, dy, center.lat, center.lon);
    coords.push([lon, lat]);
  }
  return {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [coords] },
  };
}

function SensorHud({
  accel, gyro, mag, gps, gpsStatus, fused, headingDeg, uncertaintyM,
  modelStatus, modelVelocityKmh,
}: {
  accel: Vector3 | null;
  gyro: Vector3 | null;
  mag: Vector3 | null;
  gps: GpsFix | null;
  gpsStatus: string;
  fused: { x: number; y: number; v: number } | null;
  headingDeg: number | null;
  uncertaintyM: number | null;
  modelStatus: string;
  modelVelocityKmh: number | null;
}) {
  return (
    <View style={styles.hud} pointerEvents="none">
      <Text style={styles.hudTitle}>DHRUVA — live speed model + fusion</Text>
      <HudRow label="accel (m/s²)" v={accel} />
      <HudRow label="gyro (rad/s)" v={gyro} />
      <HudRow label="mag (μT)" v={mag} />
      <Text style={styles.hudLine}>
        gps [{gpsStatus}]{' '}
        {gps
          ? `${gps.lat.toFixed(5)}, ${gps.lon.toFixed(5)} ±${gps.accuracy.toFixed(0)}m`
          : ''}
      </Text>
      <Text style={[styles.hudLine, styles.hudDivider]}>
        speed model [{modelStatus}]{' '}
        {modelVelocityKmh != null ? `${modelVelocityKmh.toFixed(1)} km/h` : ''}
      </Text>
      <Text style={styles.hudLine}>
        heading: {headingDeg != null ? `${headingDeg.toFixed(1)}°` : '—'}
      </Text>
      <Text style={styles.hudLine}>
        fused x,y: {fused ? `${fused.x.toFixed(1)}m, ${fused.y.toFixed(1)}m` : '—'}
      </Text>
      <Text style={styles.hudLine}>
        fused v: {fused ? `${(fused.v * 3.6).toFixed(1)} km/h` : '—'}
      </Text>
      <Text style={styles.hudLine}>
        confidence radius: {uncertaintyM != null ? `${uncertaintyM.toFixed(1)}m` : '—'}
      </Text>
    </View>
  );
}

function HudRow({ label, v }: { label: string; v: Vector3 | null }) {
  return (
    <Text style={styles.hudLine}>
      {label}: {v ? `${v.x.toFixed(2)}, ${v.y.toFixed(2)}, ${v.z.toFixed(2)}` : '—'}
    </Text>
  );
}

function AppContent() {
  const insets = useSafeAreaInsets();
  const state = useDhruvaFusion();
  const { fusedLatLon, uncertaintyM } = state;

  const confidenceCircle = fusedLatLon
    ? makeConfidenceCirclePolygon(fusedLatLon, Math.max(uncertaintyM ?? 5, 2))
    : null;

  return (
    <View style={styles.container}>
      <MapLibreMap style={styles.map} mapStyle={PLACEHOLDER_MAP_STYLE}>
        <MapLibreCamera
          center={fusedLatLon ? [fusedLatLon.lon, fusedLatLon.lat] : [0, 0]}
          zoom={DEFAULT_ZOOM}
          duration={300}
        />
        {confidenceCircle && (
          <GeoJSONSource id="dhruvaConfidenceSource" data={confidenceCircle}>
            <Layer
              id="dhruvaConfidenceFill"
              type="fill"
              source="dhruvaConfidenceSource"
              paint={{ 'fill-color': '#1D5FA8', 'fill-opacity': 0.18 }}
            />
            <Layer
              id="dhruvaConfidenceOutline"
              type="line"
              source="dhruvaConfidenceSource"
              paint={{ 'line-color': '#1D5FA8', 'line-width': 1.5, 'line-opacity': 0.6 }}
            />
          </GeoJSONSource>
        )}
        {fusedLatLon && (
          <Marker lngLat={[fusedLatLon.lon, fusedLatLon.lat]} anchor="center">
            <View style={styles.positionDot} />
          </Marker>
        )}
      </MapLibreMap>
      <View style={[styles.hudContainer, { top: insets.top + 8 }]}>
        <SensorHud {...state} />
      </View>
    </View>
  );
}

function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" />
      <AppContent />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  map: { flex: 1 },
  positionDot: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: '#1D5FA8',
    borderWidth: 2,
    borderColor: '#FFFFFF',
  },
  hudContainer: {
    position: 'absolute',
    left: 8,
    right: 8,
  },
  hud: {
    backgroundColor: 'rgba(20, 28, 40, 0.85)',
    borderRadius: 10,
    padding: 12,
  },
  hudTitle: {
    color: '#fff',
    fontWeight: '600',
    marginBottom: 6,
    fontSize: 13,
  },
  hudLine: {
    color: '#DDE1E8',
    fontFamily: Platform.select({ android: 'monospace', ios: 'Menlo' }),
    fontSize: 12,
    marginBottom: 2,
  },
  hudDivider: {
    marginTop: 6,
    paddingTop: 6,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(255,255,255,0.2)',
  },
});

export default App;
