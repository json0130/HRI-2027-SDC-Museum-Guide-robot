# Progress log

## step F (build): decision log, three modes, simulator  *(branch `museum-visual-profile`)*

**Built:** `decision_log.py` (JSONL per visit; `visit_id` = random 12 hex chars; `t` = seconds since visit start; whitelisted fields only, topics as closed category counts, never labels or speech; `suspended_writes` cumulative),
`simulate.py` + `config/museum_sim.yaml`. Modes: `fixed` (neutral opening, nothing else), `prior-only` (opening from the age-band prior, nothing else), `profile` (prior + guarded extraction + profiling policy with small steps/reversals).
Scenarios: child, adult, adult confidently misread as child, quiet, chatty, topic-dislike (truth neutral on every style dimension), and three swap variants (quiet+chatty pair): detected (ids swap for 4 turns, `track_conf` drops, writes suspended),
silent (same swap, `track_conf` does not drop), silent-persistent (swap from turn 8 to the end). Fake LLM with a noise knob {0, .1, .2, .3}: per item flip direction (guards cannot catch), invent a quote / low confidence / off-vocabulary
(the real guards drop these), plus a spurious topic quoting a real word. Age misread rate is a swept parameter {0, .1, .2, .3} (plus one forced misread scenario). Paired design: each random draw is keyed by (seed, scenario, person, turn, purpose),
so a visitor says and does the same thing in every mode. Metrics: wrong openings; wrong adaptations as two numbers (opposed to truth / made when truth was neutral); reversals; questions per visitor; declined offers; turns to confident (confidence >= the adapt gate)
and confident-but-wrong; profile accuracy (style sign correct / wrong / no call, false positives on truth-0 dimensions, MAE of effective value) and like/dislike precision/recall; suspended writes. Intervals: percentile bootstrap (1000) over seeds.
Break-even: W/R at which `profile` (or `prior-only`) stops beating `fixed`, labelled a sensitivity analysis; `fixed` has net benefit 0, so the ratio is right/wrong (optionally minus a per-question cost).
**Verified:** `tests_simulate.py` (determinism, fixed/prior-only never adapt or ask, misread parameter, silent swap worse than detected, noise events caught/not by the real guards, bootstrap + break-even arithmetic, config-hash tracks effective config, log content anonymous and whitelisted, `t` and random `visit_id`);
end-to-end ephemerality: a full simulated visit (all three modes, two scenarios) in an EMPTY temp dir creates only the `.jsonl` decision logs, nothing else (no `faces.npz`, `sessions.db`, `kg_state.json`). 136 passed.
**Didn't work -> fixed (before freezing):** `DecisionLog` was built before the simulator clock state existed (NameError, only when logging); fixed. My style phrase "i love chatting as we go" also matched the fake LLM's *topic* rule and produced spurious likes (dev like-precision 0.58); the phrase was changed.
**Development used only seeds >= 10000** (unit tests, plus one 6-seed smoke run on 10000..10005 to check rendering and speed). The official run uses seeds 0..199. No threshold was changed after any run.

---

## FROZEN: thresholds and simulator parameters before the first official run  *(branch `museum-visual-profile`)*

Config hash (sha256 of the canonical JSON of all effective museum configs: appearance, initiation, profile, policy, sim): `1929eb86ac450e809b77a50bbd2fc96169f52c32d67dbb239be0fa393a3d01d5`.
Gates low 0.35 / high 0.7, min_abs_value 0.3, budget 3 turns / 4 questions, ignore-after 2, cooldown 2, max level 2, reversal x0.5; weights 0.8 / 0.5 / 0.25; min_track_conf 0.5; opening min_conf 0.7; headline noise 0.1, misread 0.1, 24 turns, 200 paired seeds (0..199).
If any value changes after this point the results are a NEW run with a new hash, labelled as such; the first run is reported as it is.

---

## follow-up 5 (step 4): tracker re-association, margin rule, track confidence, suspended writes  *(branch `museum-visual-profile`)*

**Goal:** fewer id swaps; when identity is in doubt, stop writing evidence.
**Built:** `Detection(box, emb)`; `VisitTracker` combines IoU (weight 0.3) and embedding cosine, re-associates lost tracks (`reid_ttl` 300 frames) only if best cosine >= 0.6 and beats the
runner-up (lost or live) by >= 0.1, else new id; per-track `track_conf` (ambiguous match or overlapping faces without embeddings -> 0.2, +0.2 per clean frame, new/re-associated 0.6);
ambiguous frames are not fed to the age smoother; `VisitProfile.set_track_conf/apply_tracks/is_suspended`: below 0.5 evidence writes are **dropped (not buffered), counted** (`suspended_writes`, per person; extraction results report `suspended`);
`next_move` returns none for a suspended person; `AgeBandPipeline.clear()`.
**Threshold check (what I actually found):** the embedding model is facenet-pytorch InceptionResnetV1 (vggface2), already cached. The only real embeddings available are the 8 stored views of ONE enrolled person:
same-person pairwise cosine min 0.417, mean 0.674, max 0.873, and 21% of those pairs fall below 0.6. So 0.6 will reject some true returns (it fails toward a new id, which is the cheap error). I have **no different-person faces locally**,
so the impostor side (false-merge rate) is **not measured**; I did not use published figures as if measured. Those 8 views include deliberately varied poses, so live frame-to-frame cosine should sit higher.
**Limits:** synthetic embeddings in the tests (random orthogonal vectors, mixtures for look-alikes) show the logic of the margin rule only; they cannot show how well real faces separate, relatives and siblings especially. Needs a real multi-person (and related-pair) set before trusting 0.6 / 0.1.
**Verified:** exchange-places test: IoU-only swaps ids silently (track_conf stays >= 0.6), embeddings keep them; crossing: with embeddings ids stable and conf >= 0.5, without embeddings conf drops < 0.5; re-association returns the old id, look-alike at equal cosine gets a new id, a clear winner reuses, below-threshold is new, a face resembling a LIVE track is not merged into a lost one; suspension drops/ counts/ recovers; prior not consumed by a dropped write. 121 passed.
**Didn't work -> fixed:** with equal IoU and cosine weights a "same place, different face" case tied (0.5 vs 0.5); face now outweighs position (0.3/0.7). IoU-only crossing swapped ids with margin 0.18 > 0.15, i.e. unflagged; added the overlapping-faces-without-embeddings ambiguity rule. Four of my own new tests were wrong (cosine of mixtures miscomputed) and were corrected, not the code.
**Not verified:** real cameras; real `identify_all(..., with_emb=True)` output through `Detection` (not wired anywhere in this repo).

---

## follow-up 4: confirm the FaceIdentifier guard landed  *(branch `museum-visual-profile`)*

**Why:** my follow-up-1 edit script aborted halfway on a failed string match, leaving the guards half-applied; I fixed it and the suite passed, but nothing proved the `face_id.py` guards specifically.
**Done:** the `FaceIdentifier.save` / `.load` test used `importorskip`, so a missing dependency would have skipped it silently; it is now a hard import.
**Verified (mutation check):** commenting out both `ephemeral.forbid` lines in `face_id.py` makes `test_real_face_identifier_save_and_load_refuse` fail (AttributeError, because the object has no models loaded, so the guard really is the first statement); restoring them passes. `git status` clean after commit.

---

## follow-up 3: prior hidden by default, opening-only, dropped on evidence  *(branch `museum-visual-profile`)*

**Problem:** step E exposed `prior` in every profile, where a group policy could mistake it for a preference.
**Fix:** `get_person_profile` / `get_visit_snapshot` take `include_prior=False`; when included the prior is `{band, confidence, use: "opening_only"}`, and
`VisitProfile.prior()` returns `None` once the person has any real evidence (even the weakest behaviour cue).
**Deviation from the plan:** step C said the prior "decays x0.5 per evidence item"; it is now dropped outright, so the `prior:` weight/decay config was removed (it was never used anywhere).
**Verified:** profile + API tests (default snapshot contains no prior and no band string; included prior marked opening-only; None after evidence; `dimensions` unaffected); 106 passed.
**Not verified:** nothing yet reads the prior to choose an opening (that wiring is for the museum loop, which does not exist in this repo).

---

## follow-up 2: FairFace boundary tests  *(branch `museum-visual-profile`)*

**Goal:** prove race/gender never leave the FairFace backend and that the 10-19 bin reads `unsure`.
**Built (tests only, no code change):** in `tests_appearance.py` a fake onnx session returns 7 race + 2 gender + 9 age logits.
Verified: wildly different race/gender logits give an identical `AgeProbs`; the result has only `p_child`/`p_adult`/`p_teen`; each of the 9 bins maps
to child (0-2, 3-9) / teen (10-19) / adult (20+); a 10-19 face stays `unsure` through the smoother at a full window; a source check that
`estimate` reads only `logits[9:18]`. 25 tests in that file pass.
**Not verified:** that the real model's age output is sensible (still only a valid-distribution check on random pixels).

---

## follow-up 1: ephemerality enforced  *(branch `museum-visual-profile`)*

**Problem:** step C only avoided the disk paths by convention (README said "nothing enforces that").
**Fix:** `modules/ephemeral.py`, a ref-counted switch turned on by `VisitProfile()` and off by `end()`. While on, `forbid()` raises `EphemeralViolation` from:
`SessionStore(path)` (unless `":memory:"`), `FaceIdentifier.save` and `.load`, `FirstImpression.auto_enroll` / `learn_name` (and so `meet`), and `GraphStore.save`.
With no visit active every call is a no-op, so existing modes behave as before (existing tests unchanged).
**Deviation:** this edits four files imported from ChatBox (a one-line guard each, plus the import), so they now differ from the originals.
**Verified:** `tests_ephemeral.py`: with no visit a disk SessionStore, auto-enrol and graph save still work; during a visit each refuses, no file (incl. `faces.npz`) is created,
`:memory:` SessionStore still works, normal behaviour returns after `end()`; the real `FaceIdentifier` (no models loaded) refuses `save`/`load`; overlapping visits and a double `end()` keep the counter balanced.
My own step-C round-trip test saved a graph during a visit and broke, which is the guard working; that test now serialises by hand. 94 passed.
**Not verified / limits:** this guards the persistence entry points in this repo. Any other file write (e.g. someone using `np.savez` directly) is not caught. The switch is process-wide.

---

## step E: interface for the group policy  *(branch `museum-visual-profile`)*

**Goal:** expose per-person profiles and a visit snapshot to the teammate's group policy, without aggregating.

**Built:** `modules/museum/profile_api.py` (`start_visit`, `end_visit`, `get_person_profile`, `get_visit_snapshot`; JSON-serialisable) and
`modules/museum/README.md` (contract, rules, age model note). Top-level README updated: the old line "never infers from age" is now "age band shapes only the opening move";
`proactivity` (listed in the old README) is not built; the brief's three dimensions are.

**Verified:** `modules/museum/tests_profile_api.py` (contract keys, JSON round trip, prior kept out of `dimensions`, unknown ids raise, `end_visit` wipes);
full suite 90 passed (graph_relationship + museum tests 83, `test_first_impression.py` 7 under pytest; the script itself prints all passed).
**Not verified:** that the group policy's real needs fit this shape (no consumer exists yet); `prior` is exposed in the profile dict as a labelled hint, which a consumer could misuse.

---

## step D: profiling interaction policy  *(branch `museum-visual-profile`)*

**Goal:** decide the next move toward learning a visitor, with low regret.

**Built:** `modules/museum/profiling_policy.py` (`ProfilingPolicy`, `Move`) and `config/museum_policy.yaml`. Question budget (min 3 turns apart, max 4 per
person), allow-listed wording only (choice questions first, then one open interest question), dimension = argmax (1-confidence) x usefulness, gates
low <0.35 generic / 0.35-0.7 offer / >=0.7 adapt, adaptation levels -2..+2 one step at a time with a cooldown. Declined / ignored / contradicted ->
revert that step, confidence x0.5, logged with `reversal=True`. `step_log` is an in-memory list of JSON-serialisable dicts (the evaluation/log
persistence work is deferred).

**Verified:** `modules/museum/tests_policy.py` (12 tests, all with the real `VisitProfile` and no LLM); full suite 87 passed.
**Didn't work -> fixed (tests, not policy):** two of my own tests failed at first: one expected an offer on `depth` although the argmax correctly
chose a more useful, more uncertain dimension (fixed by zeroing the other usefulness values in that test); one used contradicting evidence that left
|value| at 0.297, under the 0.3 floor (fixed with more evidence).
**Not verified:** the thresholds, budget and cooldown are untuned guesses; no evidence yet they are good. Nothing here ever ran against a human or real LLM.

---

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
