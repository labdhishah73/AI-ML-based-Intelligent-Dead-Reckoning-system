# DHRUVA — Project Context & Decision Log

Handoff document. Everything decided so far, why it was decided, and what is still open.

---

## 1. Identity

| | |
|---|---|
| Problem Statement | SIH26168 — AI-ML based Intelligent Dead Reckoning system for seamless navigation |
| Solution name | DHRUVA |
| Team | Slothdevs · Team ID F2BC5B |
| Theme / Category | Smart Vehicles / Software |
| Event | Smart India Hackathon 2026 |

**Current stage: idea submission. No code written. No models trained. No results measured.**

This matters for how everything is worded — describe the design in present tense, use "will" for anything not yet built, and never state a performance figure as achieved.

---

## 2. The problem in one paragraph

GNSS fails in tunnels, underground parking, urban canyons and dense foliage. High-end cars solve this with factory-fitted INS wired to wheel sensors. Most vehicles in India — trucks, older cars, two-wheelers — have only a driver's smartphone on the dashboard. The task is to keep lane-level position using a bare phone: no OBD-II, no wheel speed feed, no connection to the vehicle.

The hard part is that MEMS IMU errors compound. Integrating a biased accelerometer once gives linearly growing velocity error; integrating again gives positional error growing with the square of time. Unmitigated, the estimate is useless within seconds.

### Mandated benchmarks

- Dead reckoning drift **under 10% of distance travelled** (e.g. <100 m over 1 km at 60 km/h)
- **10 Hz** position updates on mobile
- **~200 Hz** on the edge engine with FOG-grade IMU input

### Required deliverables

1. Mobile application
2. **Edge-deployable software engine** — same models running standalone for external IMU data, not phone-only

---

## 3. Architecture

A strapdown inertial navigation system with GNSS aiding, running entirely on-device. Four stages between sensors and screen.

### Stage 1 — Alignment & calibration · *not trained*

Resolves the body-to-vehicle frame transformation, so the phone can be mounted at any angle.

- **Madgwick filter** (pronounced MAJ-wick) fuses accel/gyro/magnetometer into an orientation quaternion. Gravity constrains pitch and roll.
- **PCA over forward acceleration** recovers the remaining yaw offset — a car only accelerates along one axis, so the dominant horizontal acceleration axis *is* the vehicle's forward axis.
- GNSS course-over-ground assists yaw estimation when signal exists (slow side-channel, not per-sample).

*Kept classical deliberately: orientation from gravity and heading is solvable geometry, and an interpretable filter defends better than an unexplained network.*

**Known weakness:** Madgwick assumes the accelerometer reads mostly gravity, which is least true during hard driving. Mitigate by low-passing the input and making β adaptive.

### Stage 2 — Speed & vibration filter · **TRAINED — the main model**

Supplies the **velocity observation**. This is the load-bearing component: it replaces the speedometer, which is what satisfies the "no vehicle connection" constraint.

- **CNN** extracts local vibration patterns from windowed IMU data — pothole jolts, engine harmonics, road texture.
- **LSTM** carries state across the sequence, enforcing that velocity cannot jump discontinuously.
- Dense layer regresses to scalar forward velocity.

**Key insight:** this *recognises* speed from the vibration signature rather than integrating acceleration. Per-window regression means errors do not accumulate across windows — no snowballing.

**Known weakness:** generalisation. The model learns signatures present in training data; unfamiliar vehicles or road surfaces degrade it. The problem statement supplies extra datasets at evaluation precisely to test this.

### Stage 3 — GNSS+INS fusion · **partly trained**

An **Unscented Kalman Filter**. The only component holding a unified position estimate.

**State vector:** position, velocity, attitude, **IMU bias terms**.

Including bias states means GNSS-available intervals actively calibrate the sensor model for the next outage. Good-signal driving improves tunnel performance — a non-obvious point worth making aloud.

**Why UKF over EKF:** avoids hand-derived Jacobians on a quaternion state; handles large covariance growth during long outages better than local linearisation; named explicitly in the problem statement.

**Why not a factor graph:** likely more accurate, but cost is unjustifiable for 10 Hz on-device inference.

**Coupling:** loosely coupled — consumes the GNSS position/velocity solution, not raw pseudoranges. Tight coupling would help under partial satellite visibility but Android does not expose raw measurements reliably across devices.

**The trained part:** a small network predicts measurement noise covariance from GNSS quality indicators (satellite count, reported accuracy, SNR). Replaces hand-tuned fixed covariance, so multipath-degraded urban fixes are trusted less than open-highway fixes.

### Stage 4 — Map matching & kinematic constraints · *not trained*

- **HMM over the OpenStreetMap road graph**, decoded with **Viterbi** (vi-TER-bee). Finds the most likely *sequence* of road segments, not the most likely individual snap — which is what prevents teleporting between parallel roads.
- **Non-holonomic constraints** rule out physically impossible motion.

This is the component that closes the gap to the <10% drift benchmark. The IMU alone probably won't hit it; the IMU constrained to a road graph will.

### Stage 5/6 — Deficit handler and UI

**Deficit handler** is arbitration logic, not an algorithm: monitors satellite count and reported accuracy, gates whether GNSS reaches the update step.

**UI** renders position at 10 Hz over offline MapLibre tiles, with Reanimated interpolating between updates. Two features:
- **Confidence radius** sized by the filter's covariance. Strong because the number already exists as a by-product of filtering — you're surfacing it, not inventing it. Google Maps shows a confident dot while guessing; this is more honest.
- **Uninterrupted voice guidance** — route and road graph are local, so turn-by-turn continues underground.

---

## 4. THE architectural point

**GNSS appears only in the measurement update. It is absent from the process model.**

Consequences:
- Stages 1 and 2 behave identically with or without signal
- An outage degenerates to prediction-only propagation
- No reconfiguration, no mode switch, no re-initialisation

The seamlessness is **structural**, not an engineered handover. This is the strongest technical claim in the pitch and should never be cut.

**Framing to use:** one continuously running inertial system that accepts GNSS corrections when available — *not* two modes with a switch between them.

> ⚠️ The slide 3 diagram labels "GNSS Mode" and "Dead Reckoning Mode" as two boxes, which sits awkwardly against this framing. Consider relabelling to "GNSS-aided" and "Inertial only".

---

## 5. Technical stack

**On-device app** — React Native (bare CLI), MapLibre GL with offline OSM tiles, react-native-sensors, Reanimated, SQLite, MBTiles

**Trained models (2)** — CNN-LSTM speed estimator; adaptive noise network for fusion trust. TensorFlow → TFLite, int8 post-training quantization

**Classical engines** — Madgwick + PCA alignment; Unscented Kalman Filter; HMM map-matching with NHC

**Training** — Python, TensorFlow/Keras, NumPy, Pandas, filterpy, OSMnx, Google Colab / Kaggle

**Backend** — Node.js, Express, REST, JWT, MongoDB Atlas, Render. **Not in the positioning path** — auth, trip persistence, model distribution only.

**Zero paid API keys. Zero network calls during navigation.**

### Decisions and rejected alternatives

| Chose | Over | Why |
|---|---|---|
| React Native | Flutter | Mobile app needs only 10 Hz (the 200 Hz figure applies to the edge engine, not the phone). At that rate the framework is not the bottleneck, and team velocity is. Flutter's rendering advantage doesn't cash in here. |
| React Native | Native Kotlin/Swift | Native gives tighter sensor and background-execution control, but costs a full UI rewrite in an unfamiliar language. RN allows dropping *just* the sensor layer into a native module later if sampling reliability becomes the limit. |
| MapLibre + offline OSM | Google Maps SDK | Removes paid keys and the connectivity dependency the whole project argues against. |
| MongoDB | PostgreSQL + PostGIS | Trip data is document-shaped, pairs cleanly with Node. PostGIS is the stronger answer *if* trajectory/geospatial queries become central — know this as the fallback answer. |
| TensorFlow | PyTorch | TFLite export is native; avoids a PyTorch→ONNX→TFLite conversion step under deadline. |

---

## 6. Dataset — IO-VNBD

Say it as letters: **I-O-V-N-B-D**.

| | |
|---|---|
| Total files | 564 |
| **S files** | 241 — phone-side. Accel, gyro, magnetometer, GNSS. **The question.** |
| **V files** | 323 — vehicle-side. Wheel speeds, speedometer, steering angle, gear, RPM. **The answer key.** |
| Size | 30–40 GB full set |
| Column layouts | 6 variants across files |

### Gotchas

- Some files name gyro axes **X/Y/Z**, others **Yaw/Pitch/Roll**. Not interchangeable — mismatched axes silently poison training data.
- Too large for memory. Use a streaming loader (`tf.data.Dataset` from a generator), reading batches off disk.
- **Data leakage risk:** V-file speed is ground truth for training only. It must never reach the model at inference or during simulated blackout evaluation. Results would be meaningless. Be ready to state this unprompted — it is exactly what a sharp examiner probes.

---

## 7. Open items and known gaps

### 🔴 Missing screening requirement

The problem statement requires: *"preliminary AI models and the results of the position plot inferenced from the subset of IO-VNBD dataset as part of their proposals."*

**Neither exists yet.** Highest-leverage remaining work. Even a non-ML baseline — integrate the raw IMU from one file, plot the drifting trajectory against the true GPS path — would satisfy "something", and is a few hours of Python.

### 🟡 Edge-deployable engine underspecified

Required as a co-deliverable. Currently appears nowhere in the deck except one line of the script. The answer is short — the same trained models run standalone in Python for external IMU input, sensor-agnostic by design — but it needs to be visible somewhere.

### 🟡 Sensor availability gap

Not every phone has every sensor. Tiered degradation:

- **Accelerometer** — universal. Safe to assume.
- **Magnetometer** — often absent on budget Android. Barely a loss: it's unreliable inside a steel car body anyway, and heading comes from PCA + GNSS course rather than magnetic north.
- **Gyroscope** — **the real dependency.** Absent on many entry-level Android devices, exactly the target market. Without it, attitude cannot be propagated.

**Tier 3 fallback (accel only):** route-constrained dead reckoning — propagate along the OSM road polyline using speed from the accelerometer, heading inherited from the road. Works on unbranched stretches (tunnels, underpasses — the primary use case), degrades at junctions.

Detect via `SensorManager.getDefaultSensor()` returning null at startup.

*Do not quote a percentage for gyro-less phones without verifying it.*

### 🟡 Slide 3 image inaccuracies

The left stack panel is a rasterised image and cannot be text-edited:
- Says **"Kalman Filter"** — should be **"Unscented Kalman Filter"** (the diagram beside it already says Unscented, so the deck is inconsistent, not wrong)
- Omits the adaptive noise network and NHC from the stack list (both appear in the diagram)
- Does not name IO-VNBD in the stack list

The garbled "cms switch" label has been covered over. Remaining fixes require regenerating the image.

### 🟠 NHC placement — correction worth knowing

In the navigation literature, non-holonomic constraints are normally implemented as **pseudo-measurements inside the Kalman filter** — zero lateral and zero vertical velocity in the body frame, fed as observations with small noise. That is **stage 3**, not stage 4.

The diagram places NHC in stage 4 alongside map matching, following the problem statement's own phrasing. Not wrong, but an examiner asking "how do you apply the NHC?" expects the pseudo-measurement answer.

**Safe answer covering both:** *"We apply it as a pseudo-measurement in the filter — zero lateral and vertical velocity in the body frame — and the map-matching layer additionally rejects transitions requiring impossible motion."*

---

## 8. Q&A positions

**"Where is the AI? This looks like classical engineering."**
Trained models exactly where classical methods fail — velocity with no speedometer (no closed-form equation) and noise covariance (conventionally guessed once). Classical everywhere the problem is already solved, because a proven method beats a network we'd have to justify. *Better than claiming everything is AI.*

**"Why not one end-to-end neural network?"**
Needs far more data than IO-VNBD provides, wouldn't transfer across the sensor grades the problem statement requires (phone MEMS *and* FOG), and gives no uncertainty estimate — which we need for the confidence radius.

**"How do you meet the drift benchmark?"**
Not with sensors alone. The road-network constraint is what closes the gap.

**"What's the hardest engineering problem?"**
Re-acquisition on tunnel exit. After a long outage the first fix may disagree by tens of metres; applying it at full weight makes the marker jump — the exact defect we claim to fix. The filter handles it *if* covariance bookkeeping is right: large uncertainty means it eases across rather than snapping. The adaptive noise network also down-weights the first weak fix.

**"What results do you have?"**
Architecture and stack are settled; models are the next step; IO-VNBD provides matched phone and vehicle data to train and validate against. **Do not improvise a number.**

**"What stops the learned velocity drifting?"**
Per-window regression, not integration — errors don't accumulate across windows. Filter bias states absorb systematic offset. Map matching bounds cross-track error independently.

---

## 9. Terminology quick reference

| Term | Plain meaning |
|---|---|
| Dead reckoning | Position from last known point + speed + heading + time, no external reference |
| IMU / MEMS | Motion sensor package / the cheap phone-grade class of them |
| GNSS | Umbrella for GPS, Galileo, NavIC. **Not** the internet — receive-only, works in airplane mode |
| Drift | Error accumulating over time |
| Sensor bias | Constant offset in readings; harmless alone, damaging once integrated |
| Quadratic error growth | Double-integrating a biased signal → error ∝ t² |
| Covariance | The filter's own uncertainty estimate. What the confidence radius displays |
| Predict / update | Filter's two half-steps: propagate from IMU (uncertainty grows) / correct from measurement (uncertainty shrinks) |
| Multipath | Satellite signal arriving via building bounces; wrong position. Urban canyons |
| Viterbi | Finds the single most likely *sequence* of hidden states from noisy observations |
| Non-holonomic | Constrains *how* you can move, not *where* you can end up. Car can't slide sideways |
| int8 quantization | Weights stored as 8-bit ints instead of float32 — ~4× smaller, faster, <1% accuracy cost |
| MBTiles | Single file of pre-rendered map tiles, read from local storage |
| Loosely coupled | Fusing the GNSS position/velocity solution rather than raw pseudoranges |

---

## 10. Artefacts produced

| File | Purpose |
|---|---|
| `dhruva_explained.html` | Full plain-language explainer — 15 sections, both flowcharts, complete glossary |
| `dhruva_ppt_final.pptx` | The deck. Benefits panel added to slide 5; text corrections across slides 1, 2, 4, 6 |
| `dhruva_slide3_script_v2.docx` | Slide 3 speaking script, ~330 words, 2:05–2:15, with cut list and Q&A |
| `idr_technical_stack.html` | Earlier stack reference with per-decision rationale |

---

## 11. Communication rules for this project

- **Never state an unmeasured result.** Nothing is built. Benchmarks are targets being designed to.
- Say **Unscented Kalman Filter**, not Kalman Filter.
- Say **GNSS**, not GPS.
- Say **velocity observation**, not speed prediction — it enters the filter as an observation, and that framing is why the architecture holds.
- **Two** trained models. Not four, not six. Precision here reads as competence.
- If a term can't survive a follow-up question, don't use it.
