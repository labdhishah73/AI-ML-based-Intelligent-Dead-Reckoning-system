/**
 * Manual Jest mock for react-native-fast-tflite — same reason as the other
 * native-module mocks in this directory: it registers a Nitro/TurboModule
 * (via react-native-nitro-modules) that only exists inside an actual
 * running app binary, which Jest's test environment never has.
 *
 * loadTensorflowModel() resolves to a fake model whose runSync() returns a
 * zeroed output buffer — enough for App.test.tsx's "renders correctly"
 * smoke test, which never asserts on an actual predicted velocity.
 */
function runSync() {
  return [new Float32Array([0]).buffer];
}

function run() {
  return Promise.resolve(runSync());
}

const fakeModel = {
  delegates: [],
  inputs: [],
  outputs: [],
  runSync,
  run,
};

module.exports = {
  loadTensorflowModel: () => Promise.resolve(fakeModel),
  useTensorflowModel: () => ({ state: 'loading', model: undefined }),
};
