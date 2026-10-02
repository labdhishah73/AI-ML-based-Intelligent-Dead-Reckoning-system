/**
 * Standalone sanity check for SimpleFusionFilter — not a Jest suite, just a
 * quick script to confirm the filter behaves sensibly before the
 * Integration Engineer wires it into the live app. Run with:
 *   npx tsc src/fusion/selfTest.ts src/fusion/kalmanFilter.ts --outDir .tmp_selftest --module commonjs --target es2019
 *   node .tmp_selftest/selfTest.js
 */
import { SimpleFusionFilter } from './kalmanFilter';

let anyFailed = false;
function assert(cond: boolean, msg: string) {
  if (!cond) {
    console.error('FAIL:', msg);
    anyFailed = true;
  } else {
    console.log('ok:', msg);
  }
}

// Scenario 1: driving due east at a constant 10 m/s for 10s with no GPS —
// pure dead reckoning should land close to (100, 0).
{
  const f = new SimpleFusionFilter();
  const dt = 0.1;
  for (let i = 0; i < 100; i++) {
    f.predict(0 /* heading = due east */, dt);
    f.updateVelocityObservation(10, 1.0);
  }
  const s = f.getState();
  console.log('scenario 1 final state:', s);
  assert(Math.abs(s.x - 100) < 5, `x should be near 100, got ${s.x.toFixed(2)}`);
  assert(Math.abs(s.y) < 1, `y should stay near 0, got ${s.y.toFixed(2)}`);
  assert(Math.abs(s.v - 10) < 1, `v should converge near 10, got ${s.v.toFixed(2)}`);
}

// Scenario 2: uncertainty grows during a GNSS outage, then shrinks the
// instant a GPS correction arrives — this is the confidence-circle behavior
// the UI depends on.
{
  const f = new SimpleFusionFilter();
  const dt = 0.1;
  const uncertaintyBefore = f.getPositionUncertaintyMetres();
  for (let i = 0; i < 50; i++) {
    f.predict(0, dt);
    f.updateVelocityObservation(10, 1.0);
  }
  const uncertaintyDuringOutage = f.getPositionUncertaintyMetres();
  assert(
    uncertaintyDuringOutage > uncertaintyBefore,
    `uncertainty should grow with no GPS: before=${uncertaintyBefore.toFixed(2)} during=${uncertaintyDuringOutage.toFixed(2)}`,
  );

  f.updateGpsPosition(f.getState().x, f.getState().y, 3.0);
  const uncertaintyAfterFix = f.getPositionUncertaintyMetres();
  assert(
    uncertaintyAfterFix < uncertaintyDuringOutage,
    `uncertainty should shrink after a GPS fix: during=${uncertaintyDuringOutage.toFixed(2)} after=${uncertaintyAfterFix.toFixed(2)}`,
  );
}

// Scenario 3: re-acquisition doesn't snap — a single GPS fix disagreeing by
// 80m after a long outage should pull the estimate partway, not all the way,
// on the very first correction.
{
  const f = new SimpleFusionFilter();
  const dt = 0.1;
  for (let i = 0; i < 300; i++) {
    // long outage - 30s of dead reckoning growing uncertainty
    f.predict(0, dt);
    f.updateVelocityObservation(10, 1.0);
  }
  const beforeFix = f.getState();
  const trueX = beforeFix.x + 80; // GPS disagrees by 80m
  f.updateGpsPosition(trueX, beforeFix.y, 5.0);
  const afterFix = f.getState();
  const movedBy = afterFix.x - beforeFix.x;
  console.log(`scenario 3: disagreement=80m, filter moved by=${movedBy.toFixed(1)}m`);
  assert(movedBy > 0 && movedBy < 80, `should ease in partway, not snap fully: moved ${movedBy.toFixed(1)}m of 80m`);
}

console.log(anyFailed ? '\nSelf-test complete — FAILURES ABOVE.' : '\nSelf-test complete — all passed.');
