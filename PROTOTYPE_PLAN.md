# DHRUVA — Prototype Plan (target: demo-ready by end of day, Sept 7 2026)

Today is Sept 4. That gives three working days (Sept 5, 6, 7) after today's setup.
Team: 6 people, roles not yet assigned. This plan assigns function-based tracks —
map real people to them based on actual skills, not job titles.

**Guiding rule carried over from the project context doc: never claim a number
you haven't measured on your own test run.** Whatever drift/accuracy the demo
actually shows on Sept 7 is what you say in the room.

---

## 0. Scope cuts from the full architecture — and why

The full design (see `DHRUVA_PROJECT_CONTEXT.md`) has 6 stages, 2 trained models,
a bias-state UKF, and HMM/Viterbi map matching. That is a post-prototype target,
not a 3-day build. Cuts made for this prototype, each with a stated reason and
an explicit "this is what we say if asked":

| Full design | Prototype version | If a judge asks why |
|---|---|---|
| CNN-LSTM speed model | **Kept as-is.** This is the load-bearing model — everything else is buildable without it, this isn't. | N/A — this is the one piece that must not be cut |
| Adaptive GNSS noise network (trained) | Rule-based thresholds on satellite count / reported accuracy | "The learned version is designed and dataset-ready; for the prototype we use hand-tuned thresholds so we could spend the time budget on the load-bearing speed model instead" |
| UKF with IMU bias states | Simplified Kalman filter: position, velocity, heading — **no bias states in v1** | "Bias estimation is architected — state vector has the slots — we're validating the core fusion loop before adding it" |
| HMM + Viterbi map matching | Nearest-road snapping + heading-consistency check, run only over the demo area's clipped OSM extract | "Full sequence-decoding map matching is the next milestone; for the prototype scope we show the constraint qualitatively" |
| Non-holonomic constraint as KF pseudo-measurement | Simple heading-delta sanity check post-filter | Same as above — mention the correct textbook placement (stage 3, pseudo-measurement) even though the prototype implements it as a post-check |
| Voice guidance | Stub / best-effort TTS on canned route steps | Not core to the drift claim — cut first if time runs out |

**Do not cut:** the speed model, the live sensor pipeline, the offline map, and
the "GNSS joins only as a correction, dead reckoning never stops" behavior —
that's the actual thesis of the pitch.

---

## 1. Role split (map your 6 people onto these)

1. **ML Lead** — dataset windowing, CNN-LSTM training, export to TFLite.
2. **ML Support / Data Engineer** — dataset loading, S/V pairing, axis-naming
   cleanup (the X/Y/Z vs Yaw/Pitch/Roll gotcha), the required position-plot
   baseline (raw IMU integration vs true GPS).
3. **Mobile App Lead** — React Native scaffold, MapLibre + offline MBTiles,
   screen/UI, confidence circle.
4. **Sensor & Fusion Engineer** — Madgwick orientation, the simplified Kalman
   filter, sensor stream plumbing (accelerometer/gyro/mag/GPS into JS).
5. **Integration Engineer** — gets the trained TFLite model running inside the
   RN app, wires fusion output to the map at ~10 Hz, on-device testing.
6. **Infra / Demo / Docs** — OSM extract for the demo area, the standalone
   "edge-deployable engine" script (same model, plain Python, CSV in → plot
   out — this doubles as your edge deliverable AND the missing screening
   requirement), demo script + fallback recording, deck updates.

Two people can float between 2↔6 and 3↔4 depending on how the actual skill
split lands — reassign after a 15-minute skills check today.

---

## 2. Immediate blocker: the dataset

Nothing on the ML track starts until the dataset is in hand. **Send it now** —
ideally as a shared drive link or a subset (2–3 matched S+V file pairs is
enough to start; don't wait for the full 30–40GB set).

**Status as of Sept 4:** only the two schema reference docs (`columns in
dataset (1).docx`, `IO-VNBD_Column_Reference_Guide.docx`) are in the project
folder — no actual CSV data yet. Those docs were enough to reverse-engineer
the full column schema and pre-build the data-loading layer (`ml/schema.py`,
`ml/data_loader.py`, `ml/baseline_position_plot.py` — see `ml/README.md`).
It's wired up and ready; it just needs `IOVNBD_ROOT` pointed at the real
files the moment they land.

Two scope decisions already baked into that code, so the ML track doesn't
burn Day 1 rediscovering them:
- **Train only on the "Synchronised V abd S datasets" tree.** The
  "Unsynchronised" tree has no row-level time alignment between S and V
  files — using it needs timestamp interpolation, which isn't worth building
  under this deadline.
- **Train only on Structure 1 S files** (Yaw/Pitch/Roll gyro labeling — 72 of
  the 158 Structure-1 files sit in the synchronised tree already). Structures
  3/4 label the same columns X/Y/Z / Azimuth, and whether those are the same
  physical axes as Structure 1 is unverified — mixing them in blind is
  exactly the "mismatched axes silently poison training data" risk flagged
  in `DHRUVA_PROJECT_CONTEXT.md`.

Also decide **today**: what is the demo itself?
- **Recommended:** live phone carried during an actual short walk/drive around
  a real area, with a real GNSS-blocking spot (underground parking, a
  covered walkway, or physically shielding the antenna) to show the dead-
  reckoning-only segment and the re-acquisition behavior.
- **Fallback (build this too, don't skip it):** a canned/replayed sensor log
  driving the same UI, so a live-demo hiccup in front of judges doesn't sink
  the whole thing. Record this fallback on Day 2, not Day 3.

---

## 3. Day-by-day

### Day 0 — today, Sept 4 (evening)
- Get the dataset from you into the ML tracks' hands.
- 15-min skills check → finalize who's on which track.
- Repo set up (git, shared), RN project scaffolded (empty shell), Python/Colab
  notebook created, package installs: `react-native-sensors`, MapLibre RN
  bindings, `filterpy`, TensorFlow, `osmnx`.
- Pick and clip the demo area's OSM extract; pick the physical demo route.

### Day 1 — Sept 5
- **ML:** clean 2–3 matched S/V file pairs, build the windowed streaming
  loader, train a first-pass CNN-LSTM, get a loss curve and predicted-vs-true
  speed plot (this is also literally the missing screening deliverable).
- **Fusion:** implement Madgwick orientation (use a known-good open-source
  implementation, don't write it from scratch), stub the KF structure with a
  placeholder speed input so the interface is fixed early.
- **Mobile:** RN scaffold running on a real device, MapLibre rendering the
  offline demo-area tiles, raw sensor streams (accel/gyro/mag/GPS) flowing
  into JS and logged to console — no fusion yet, just plumbing.
- **Infra/Docs:** OSM extract finalized, README + role assignments written
  down, deck skeleton started, begin the standalone edge-engine script shell.

**End-of-day checkpoint:** speed model trains and produces a plot; sensors
stream into the app; orientation filter runs on canned data. If any of these
three isn't true, that's tomorrow's first priority, not new features.

### Day 2 — Sept 6
- Export the speed model to TFLite (int8), integrate into the RN app via a
  TFLite binding, confirm on-device inference on a live sensor window.
- Wire: orientation (Madgwick) + speed (TFLite model) + GPS → simplified KF →
  position stream → map marker, targeting as close to 10 Hz as achievable.
- Add the road-snapping + heading-consistency check on top of the fused
  position, scoped to the clipped demo-area OSM extract only.
- Add the confidence circle (size it off the KF's own covariance — don't
  invent a separate number).
- **Record the fallback demo video/log now**, not on Day 3.
- Finish the standalone edge-engine script (CSV in, position plot out) using
  the same trained model — this is your edge-deployable deliverable.

**End-of-day checkpoint:** a dot moves on the offline map in response to a
real live phone carried around, GPS correction visibly happens, and there's a
recorded fallback in hand.

### Day 3 — Sept 7 (demo day)
- Morning: full field test on the real demo route, including the GNSS-blocked
  segment. Measure and write down actual drift — do not estimate it.
- **Feature freeze by midday.** Bug fixes only after that.
- Finalize deck using the correction notes already in
  `DHRUVA_PROJECT_CONTEXT.md` (§7: UKF naming on slide 3, NHC placement
  answer, "GNSS-aided" vs "Dead Reckoning Mode" labeling).
- Full rehearsal: live demo first, fallback video ready to switch to
  immediately if sensors misbehave in the room.

---

## 4. Risks worth naming now, not discovering on Sept 6

- **TFLite-in-RN integration is the most likely single point of failure** —
  give it to whoever is most comfortable with native module bridging, and
  start it Day 2 morning, not afternoon.
- **Live GNSS-blocking in a real demo is unpredictable** — this is exactly
  why the recorded fallback isn't optional.
- **Sensor sampling rate varies by phone model** — test on the actual demo
  phone early, not on whatever's convenient on Day 1.
- **CNN-LSTM won't be well-tuned in one day of training** — that's fine and
  expected; the plot showing it beats raw IMU integration is the deliverable,
  not a polished accuracy number.

---

## 5. What this plan deliberately does NOT try to be

This is not the final SIH submission architecture. It's a working, honest,
live demo of the thesis — dead reckoning that doesn't stop when GNSS does —
built around the one model that must exist (the speed estimator) plus just
enough classical scaffolding to make a real dot move on a real offline map.
The bias-state UKF, learned noise network, and HMM map matching are the
documented next milestones, not missing work.
