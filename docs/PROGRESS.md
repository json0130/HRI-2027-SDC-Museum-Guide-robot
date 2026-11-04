# Progress log

## step C: visit-scoped profile in the KG  *(branch `museum-visual-profile`)*

**Goal:** per-person, per-visit likes, dislikes and style (`talk_level`, `depth`, `pace`) with value, confidence and evidence.

**Schema (additive, `schema.py`):** `Evidence`, `StyleDimNode`, `ProfileEdge` (`visit_pref`, FAST, signed `value` in [-1,1], `confidence`,
`visit_id`, `evidence[]`). Existing types untouched; `PersonAttributeEdge` and `get_person_context` do not see the new edge, so existing
consumers are unaffected. `PreferenceEdge` (SLOW, unsigned) and `AboutEdge` (SLOW, no visit scope) were not reused: they would be long-term memory.
**Built:** `modules/museum/profile.py` (`VisitProfile`: private `InMemoryGraphStore`, never saved; noisy-OR confidence; `effective_value = value x confidence`;
visual prior held outside the KG and decaying x0.5 per real evidence; `end()` deletes persons, edges, created topics and style nodes),
`profile_extraction.py` (per-turn, injected `llm_fn`, closed vocab, `CONFIDENCE_MIN`, quote must occur in the utterance, parse/shape failure applies nothing),
`config/museum_profile.yaml`.

**Verified:** `modules/museum/tests_profile.py` (old-format `kg_state` loads, new types round-trip, weights, contradiction, prior never in the KG,
guards with a fake LLM, wipe at end, no files created); full suite 75 passed.
**Not verified:** real LLM extraction quality; `SessionStore`/RAG are simply not used by museum code (nothing enforces it beyond the museum modules not importing them).
**Deviation:** extraction drops invalid items individually (as `kg_extraction` does) and applies the survivors together; a parse/shape failure applies nothing.

---

## step B: initiation policy table  *(branch `museum-visual-profile`)*

**Goal:** map (age band, confidence, group size) to opening parameters, robot-agnostic.

**Built:** `modules/museum/initiation.py` (`OpeningParams`, `choose_opening`), `config/museum_initiation.yaml`
(rows `<neutral|child|adult>_<solo|group>`, missing fields inherit `default`). `unsure`, confidence < `min_conf` (0.7) or
`use_prior=False` always select the neutral row. Band only picks the row; it is never in the wording.

**Verified:** `modules/museum/tests_initiation.py` passed; full suite 58 passed. A test checks every row has a choice question and
no age/appearance wording.
**Didn't work -> fixed:** my age-word regex flagged "look around" in a child row; the regex was too broad (changed to "you look"),
the wording was fine.
**Not verified:** that the child/adult wording suits real visitors; Pepper mapping of the parameters.

---

## step A: age-band estimator + worker  *(branch `museum-visual-profile`)*

**Goal:** coarse visual first impression (`child | adult | unsure`) as a weak prior for the opening move only.

**Built:** `modules/museum/appearance.py` (`AgeBandEstimator` protocol; `fake` and `fairface` backends; `PersonAgeSmoother`;
in-memory IoU `VisitTracker` giving `visitor_N`; `AgeBandPipeline`), `worker.py` (`AgeBandWorker`, `_DetectionWorker` pattern),
`config.py` (YAML loader), `config/museum_appearance.yaml`. FairFace reads only the 9 age logits; race/gender logits are discarded
inside the backend. No new pip dependency (onnxruntime already installed); model file is pinned by SHA-256 and not auto-downloaded.
The 10-19 bin counts toward neither child nor adult, so teens come out `unsure`.

**Verified:** `modules/museum/tests_appearance.py` 13 passed (smoothing, abstention, tracker, worker thread, no gender/race fields,
checksum refusal, one real FairFace forward pass on a random image against the cached model). Full suite green.

**Not verified:** FairFace age quality on real faces (the test only checks the output is a valid distribution); real camera; there is
no `webcam_loop` in this repo, so the `--museum` flag has nothing to attach to yet. IoU-only tracking can swap ids when people cross.

---

## import: first-impression logic from ChatBox-companion-robot  *(branch `feat/import-first-impression`)*

**Goal:** seed the museum-guide repo with the first-impression work (the brief's "guards from your
first-impression branch" + meet-a-stranger), copied — not moved — so ChatBox and its other branches
are untouched.

**Source:** `ChatBox-companion-robot` @ `65d36f3` (branch `feature/rolesim`), `CHATBOX_V5-dev/CHATBOX_SERVER/`.
Files are copied verbatim at their original relative paths so `modules.*` imports keep working.

| Copied verbatim | What it is |
|---|---|
| `modules/kg_extraction.py`, `modules/graph_relationship/extraction.py` | LLM proposes, deterministic guards decide (JSON parse, enum, hallucination, confidence, normalize) |
| `modules/graph_relationship/{schema,store,interactions,topics,scales,rename}.py` | minimum graph closure the guards and `rename_person` sit on |
| `modules/session_store.py` | transcript store (+ `rename_person`) |
| `modules/face_webcam/face_id.py` | `FaceIdentifier` (enroll / identify_all / rename / save) |
| `modules/graph_relationship/tests_*.py` | extraction, store, schema, topics pytest suites |

**New:** `modules/first_impression.py` — `extract_name` / `slug_name` and a `FirstImpression` class
(`next_guest_id`, `auto_enroll`, `learn_name`, `meet`). Logic lifted out of `WebcamKGLoop`
(`webcam_loop.py`, 3.2k lines) and its chat-Enter handler; `meet()` is that handler block as one call.
`test_first_impression.py` is the ported test (culture-pipeline + `kg_bridge` checks dropped, `meet()` check added).

**Verified:** `python3 test_first_impression.py` 6/6; `pytest modules/graph_relationship/tests_*.py` 29 passed.
Not run against a real camera/model (`face_id.py` needs cv2, torch, facenet_pytorch).

**Needs a decision before the museum build (not done here):**
- The face DB (`faces.npz`), the graph and `SessionStore` all *persist*. The brief says the robot
  "forgets everything when the visit ends" and uses "anonymous in-memory IDs". Auto-enrol + name learning as
  imported stores faces and names; it needs an in-memory / wipe-at-end mode, or to be restricted to
  the guard + anonymous-ID parts.
- The Library-Book-Robot repo already has the anonymous-ID face counting, FSM and session log the brief calls
  "already in the repos"; this import has not been reconciled with it.
- `webcam_loop.py` was deliberately not copied; nothing here wires `FirstImpression.meet()` into a loop yet.
