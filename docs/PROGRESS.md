# Progress log

## RESULT: first official tracker/trigger run (steps 1-4)  *(branch `museum-visual-profile`)*

**Run:** 100 seeds (0..99), config hash `01ae9489...b905fc` (matches the FROZEN entry), code at `f0bfbf1` (tree clean). Tables: `docs/results/museum_tracking_first_run.md/.json`. No threshold changed before or after; all development used seeds >= 10000. Numbers are per run (one scene), mean [95% bootstrap CI over seeds]. This is MY model of people and of face embeddings: it says nothing about real visitors or real faces.
**Findings (what the numbers show):**
1. **Passers-by who walk through the near field DO trigger.** `passer_by_near` false triggers 1.00 [1.00, 1.00] per run, and `crossing` (two people walking past each other at near range) 1.00 [1.00, 1.00]: the dwell test counts frames above `enter_ratio`, not standing still, so anyone who stays in the near zone for 20 frames (1.3 s) fires it. Far passers-by (`passer_by_far`) and far lingerers never trigger (0 and 0). Not fixed (no speed gate in the approved spec); the obvious candidate is a maximum speed in face-widths per second, which the group tracker already computes. A change would be a new run with a new hash.
2. **Child-sized faces are missed at a distance where an adult triggers:** `child_same_distance` missed triggers 1.00 [1.00, 1.00], `adult_at_distance` 0.00 [0.00, 0.00]. The documented box-size limitation, deliberately not corrected with the age band.
3. **Embeddings stop the id swaps that IoU-only makes.** Crossing: id swaps 0.00 [0.00, 0.00] with embeddings vs 3.26 [2.90, 3.64] IoU-only. Exchange of places: 0.00 [0.00, 0.00] vs 2.00 [2.00, 2.00].
4. **Write suspension catches an ambiguous crossing but never the silent swap.** IoU-only crossing: writes dropped 5.04 [4.84, 5.26], but writes still misattributed 4.60 [3.93, 5.26] (after the crossing the ids are swapped and `track_conf` has recovered). Silent exchange of places (IoU-only): writes dropped 0.00 [0.00, 0.00] (suspension never fired), writes misattributed 11.01 [10.88, 11.14] per run.
5. **A group arriving together gets one event for everyone** (`group_one_event` 1.00 [1.00, 1.00], time to engage 2.29 [2.24, 2.34] s) but a group that is already standing together waits for the group context: time to engage 5.07 [4.93, 5.23] s vs 1.32 [1.31, 1.33] s for a lone visitor (the 3 s pair evidence + 1.5 s stability gate). In `exchange_places` 0.70 [0.61, 0.79] visitors per run were not covered by the event, because the scripted occlusion starts at about the moment the group becomes known (a scenario artefact I did not anticipate; reported, not changed).
6. **Leaving after the trigger works as designed:** `leaves_after_trigger` events 1.00 [1.00, 1.00], re-greets 0.00 [0.00, 0.00], disengaged logged and the return logged, not greeted.
7. **Relatives: the first run did not stress the re-association margin.** At my synthetic relative cosine (0.8) wrong merges 0.00 [0.00, 0.00] and fragmented returns 0.00 [0.00, 0.00].
**Labelled SENSITIVITY runs (NOT the first run; seeds 100..199; override recorded in the hash and the report):** relative base-embedding cosine 0.9 and 0.95 (`docs/results/museum_tracking_sensitivity_relative_cos_*`). Wrong merges stay 0.00 [0.00, 0.00] in both; the margin rule refuses to merge look-alikes, so the returning relatives are given NEW ids instead (fragmented returns 1.84 [1.73, 1.93] at 0.9, 2.00 at 0.95, out of 2 people), versus 0.00 for unrelated people. The cost of failing toward "new person" is that returning relatives lose their continuity and, in a longer visit than these 22 s scenes (the 30 s cooldown has not elapsed), could be greeted again as new people.
**Not verified:** anything with real people, real faces or a real camera; how real relatives' embeddings score against each other (the false-merge side of the 0.6 threshold is still unmeasured); the trigger on children and in crowds.
**Process note:** added `--override` to the simulator for the sensitivity runs; the first-run results were produced before that edit (hash and commit above are the first run's).

---

## steps 1-4 evaluation: tracker/trigger simulator built, thresholds FROZEN  *(branch `museum-visual-profile`)*

**Scope:** evaluation limited to steps 1-4 (the older `simulate.py` evaluates the step 5/6 placeholders; it is frozen and untouched). New `modules/museum/simulate_tracking.py` + `config/museum_trackeval.yaml`: scripted people with ground-truth identity walk through a synthetic scene (15 fps, 640x480); per frame the REAL `AgeBandPipeline` (tracker), `VisitProfile.sync_tracks`, `GroupTracker` and `EngagementEngine` run on fake detections (box + synthetic face embedding).
**Scenarios:** crossing (with embeddings, and IoU-only), people exchanging places while occluded (silent swap; with embeddings and IoU-only), passer-by near (ratio 0.30) and far (0.12), lingering far, lingering near (a real visitor), a group of three (one child-scaled face) arriving together, an adult and a child-sized face at the same distance, leaving after the trigger then returning, relatives vs unrelated people returning in swapped places.
**Metrics:** false triggers on passers-by, missed triggers, id swaps, wrong merges by the re-association step, fragmented returns (a returning person given a new id), simulated evidence writes (total / dropped / misattributed to the wrong person), events, re-greets, time to engage, and whether a group got exactly one event. Bootstrap intervals over seeds.
**Synthetic embedding model:** one observation of a face has cosine 0.8 to that person's base embedding; relatives' base embeddings have cosine 0.8. Per-frame box jitter 3 px, size jitter 3%, each person's script shifted by a random +-0.4 s so seeds change the scene (the first version only varied embeddings, so IoU-only scenarios were identical across seeds: found by a test).
**Verified (tests, dev seeds >= 10000 only):** deterministic per seed; embeddings prevent the swaps IoU-only makes (crossing and exchange); the silent swap misattributes writes and write suspension never fires (`track_conf` stays high); an IoU-only crossing does drop writes (ambiguous); far / lingering-far never trigger; a close visitor triggers once; a group arriving together gets one event for everyone; a child-sized face misses where an adult triggers; leaving after the trigger disengages and the return is logged, not greeted; unrelated returners are never wrongly merged; the decision log is anonymous and the only file. 205 tests pass.
**Found while building, and NOT fixed (reported as found):** a passer-by walking across the NEAR field for about 3.5 s triggers an engagement: the dwell test counts frames above `enter_ratio`, not standing still, and the approved spec has no speed gate. A test documents it (`test_a_passer_by_walking_through_the_near_field_DOES_trigger_documented_weakness`). Any fix (for example a maximum speed in face-widths per second, which the group tracker already computes) would be a new run with a new hash, labelled as such.
**FROZEN before the first run:** config hash (sha256 of all effective museum configs: appearance, group, engagement, initiation, profile, trackeval) `01ae9489eb79997d96e03b78dd29aef9374ab6f436e3a062760b152ff8b905fc`. enter 0.20 / exit 0.15 / dwell 20 / min `track_conf` 0.6 / cooldown 30 s / settle 4 s / leave 3 s; group gap 1.5 x larger width, together 3 s of 10 s, arrival 4 s, stable 1.5 s, cap 6; tracker reid cosine 0.6, gap 0.1, ttl 300, stable_frames 5. Official run = seeds 0..99; all development used seeds >= 10000; any tuning will use separate seeds. The first run is reported as it is.

---

## step 4: engagement trigger  *(branch `museum-visual-profile`)*

**Goal:** start the first interaction once the visitor is close enough, robot-agnostic, without triggering on passers-by.
**Built:** `modules/museum/engagement.py` + `config/museum_engagement.yaml` (placeholders: enter 0.20, exit 0.15, dwell 20 frames, min `track_conf` 0.6, cooldown 30 s, 1 attempt per person, `group_settle_s` 4 s, `leave_s` 3 s); `EngagementEngine.update(t, frame_h, observations)` -> `EngageEvent`; `opening_for_event(event, profile)` runs `choose_opening` with the prior and group context at the moment of the event; `DecisionLog.log_event` (engage / disengaged / disengaged_returned: visit-scoped ids, group context, distance proxy, opening id, prior, `track_conf`, `suspended_writes`; no text, images, embeddings or names); the live demo now greets on the event instead of as soon as a track is stable (before engagement the reply is "please come a little closer"), marks the exchange started on the first chat message, and logs events; `tools/calibrate_engagement.py` (records the box ratio at three tape-measured distances; stores only distances and ratios).
**Group event policy (as approved):** fire when the FIRST member passes the dwell test, then wait up to `group_settle_s` for the group context; known -> one event for the whole group at once (a smaller-faced child member neither delays nor splits it); still unknown at the deadline -> fire with the people who have passed dwell so far, context unknown, plain neutral opening.
**Verified (synthetic frames, fake clock):** passer-by and far lingerer never trigger; solo known visitor fires once at the 20th frame and never again while staying; hysteresis holds between 0.15 and 0.20 and a dip below 0.15 resets; low `track_conf` or a missing node do not count; a group fires once for everyone when its context becomes known (including a member who never passed dwell); settle timeout fires with unknown context and `opening_for_event` gives the neutral solo row even with a child prior; pending cancelled if the first member leaves the zone; cooldown between events; leaving before the exchange disengages for good and a return is logged not greeted; leaving after the exchange started is not a disengagement; a 0.7x-scaled face fires later in the same approach (documented limitation, not corrected); real `GroupTracker` + engine: a pair arriving together gets ONE event with a known group of 2; live: far visitor not greeted, anonymous engage/disengaged log lines, end wipes group state; calibration summary/suggestion/recording logic. 193 passed.
**Calibration tool on the real camera:** the tool ran (it found my face at three successive 2 s captures: median ratio 0.30, 0.21, 0.13) but that was a `--no-prompt` smoke test with NO measured distances (the person was not standing at tape-measured spots), so those numbers are NOT a calibration and 0.20 / 0.15 stay unverified placeholders. The three measured distances must be recorded by someone with a tape: command in the README.
**Didn't work -> fixed:** the tool failed on first run with `No module named 'modules'` (script path); fixed by adding the repo root to `sys.path`.
**Not verified:** the trigger with real people approaching, children, crowds, or a real robot; whether 0.20 / 0.15 / 20 frames suit any real camera; whether the group settle window is long enough in practice (a group needs about 5 s from arrival to be "known", the settle window is 4 s after the first dwell pass, so a late-forming group will get the neutral opening).
**Uncertain:** `max_attempts_per_person = 1` is strict if someone leaves and returns a minute later (they are then logged as returned and not greeted); if the tracker issues a returning person a new id, the engine greets them as a new person.

---

## step 3: coarse group context  *(branch `museum-visual-profile`)*

**Goal:** the one new visual inference, kept narrow: `group_id`, `group_size`, `has_child_band_member` (true / false / unknown). No relationship labels (couple, family, parent/child, friends): that is appearance-based guessing of social facts, the class of risk the earlier culture inference was rejected for.
**Built:** `modules/museum/group.py` (`GroupTracker`, `GroupContext`), `config/museum_group.yaml` (all placeholders, untuned). Membership needs proximity AND co-movement held over time: edge-to-edge gap <= 1.5 x the LARGER of the two face widths; the two tracks first appeared within 4 s of each other; headings agree (both standing still, or both moving with cos >= 0.7; one still + one moving = a passer-by, not co-moving); close AND co-moving for >= 3 s within a 10 s window. Groups = connected components over stable tracks. Any membership change (join/leave/split) -> unknown until stable 1.5 s; a merge of two established groups of 2+ -> unknown for 10 s; a component above `max_group_size` (6) -> unknown; a lone visitor with a close, co-moving, not-yet-grouped companion -> unknown (a companion may be arriving). `has_child_band_member`: true if any member's band is child, false if all adult, unknown if any member is unsure.
Wiring: `choose_opening(..., group=ctx)` (unknown -> plain neutral solo opening and the age row is ignored too; a known group of 2+ ignores the individual age band and only uses `has_child_band_member`: true -> ONE placeholder row `group_child_present`, else `neutral_group`; a known solo visitor uses the age-band rows as before). `VisitProfile.set_group_context/group_context`: in memory like the age prior, never a KG fact, dropped once the person has real evidence, wiped by `end()`; `get_visit_snapshot/get_person_profile(include_group=False)` hides it by default.
**Verified (synthetic scenes, 10 Hz):** two standing together -> known group of 2 after the settle time; child / adult / unsure bands -> true / false / unknown; far apart -> separate; arrival gap 7 s -> not grouped; passer-by walking past a standing person -> not grouped; two people walking together -> grouped; the gap uses the larger face width (adult 100 px + child 40 px at 120 px gap is together; against the smaller width it would not be); 8 people in a row -> unknown (cap); two groups merging -> unknown (merge) then known again with size 4; a member leaving -> unknown while settling then known with size 2; a lone arrival is unknown until it settles; a group of three arriving together is never reported as a known solo first; opening wiring; group wording has no relationship or age words; hidden by default, dropped after evidence, not in the KG. 173 passed.
**Didn't work -> fixed:** (1) two people arriving together were flagged as a MERGE (unknown for 10 s), because each first looked like a known singleton before pair evidence reached 3 s; a merge now means two established groups of 2+. (2) the lone-visitor rule then reported a known solo context before the pair evidence existed (the exact case the group exists for), so a lone visitor with a close, co-moving companion is now unknown until the pair resolves. (3) `stable_s` 2.0 -> 1.5 so a group arriving together is known (about 5 s after arrival) inside the engagement settle window; a placeholder, not tuned on data.
**Uncertain / limits (placeholders):** image distance is not real distance (a camera angle can fake togetherness); people queueing side by side look together (co-movement does not rule that out); occlusions longer than the window dissolve a group; a companion arriving later than ~`stable_s` after the first member can still get a solo context; the child band feeding `has_child_band_member` is itself a weak guess, and a child's smaller box shifts the gap measure only partly (the larger width helps). No real-camera or multi-person test.
**Not verified:** anything with real people; whether `group_child_present` (or any wording) is appropriate, since step 5 is research first.

---

## rescope: steps 5-6 are provisional placeholders, F is frozen  *(branch `museum-visual-profile`)*

**Decision (rescope brief):** only steps 1-4 of the six-step flow are in scope; 5 (greeting wording, first choice question) and 6 (collecting information, building the profile) are research first.
**Done:** `modules/museum/README.md` now opens with a scope/status table and lists the provisional placeholders (initiation table, profile, extraction, policy, the conversation half of the live demo). They are frozen: bug fixes only, no new question types, templates, dimensions, thresholds or policy logic.
**Not built, as instructed:** no new Step F profile modes or profile-accuracy simulator work.
**Conflict to flag:** the earlier Step F WAS already built and run before this brief (`simulate.py`, three modes, first official run with config hash `1929eb86...`, results in `docs/results/`). I did not delete it. It is marked frozen and not extended; its numbers evaluate placeholders against my own visitor model and must not be read as evidence that the greeting, the first question or the policy works. Say if you want it removed from the branch instead.
**Also unchanged but worth stating:** the live demo still greets as soon as a stable track exists (step 4's proximity trigger is not built yet) and its conversation half uses the frozen placeholders.
**Not verified:** every placeholder. Nothing here says they work.

---

## steps 1-2: visit-scoped identity and KG node  *(branch `museum-visual-profile`)*

**Scope (from the rescope brief):** one id per visitor for the whole visit; the person is a visit-scoped KG node; nothing biometric survives the visit.
**Already built earlier (follow-up 5, not redone):** in-memory running embedding mean per track; match score = IoU (0.3) + embedding cosine (0.7); lost tracks kept `reid_ttl` = 300 frames; re-association needs cosine >= 0.6 AND a >= 0.1 margin over the runner-up (live or lost), else a new id; `track_conf` drops to 0.2 on an ambiguous match (small margin, or overlapping faces without embeddings) and recovers +0.2 per clean frame; ambiguous frames are not fed to the age smoother; below `track_conf` 0.5 profile writes are dropped (not buffered), counted, and the count is in the decision log; crossing and silent-swap tests; guard test for `FaceIdentifier.save/.load` (mutation-checked: removing the guards makes it fail), `git status` clean at 81ebcb1.
**Built now:**
- `PersonNode.visit_id` (optional, default None; old `kg_state.json` loads unchanged, tested).
- Tracker `streak` = consecutive matched frames (0 after a miss, 1 on a new or re-associated track).
- `VisitProfile.sync_tracks(results, stable_frames)` creates the `visitor_N` node only after `stable_frames` (default 5, `config/museum_appearance.yaml`) CONSECUTIVE frames, so passers-by and one-frame detections never become nodes; a broken streak does not count.
- `VisitProfile.add_cleanup`: `end()` now also runs `pipeline.clear` (wipes the in-memory face-embedding gallery and smoothers), so nothing is left that could re-identify anyone. The live demo binds it and uses `sync_tracks` instead of its own ad-hoc rule.
**Threshold check (what I found, unchanged):** embedding model = facenet-pytorch InceptionResnetV1 (VGGFace2). The only real embeddings I had were 8 views of ONE person: same-person cosine min 0.42 / mean 0.67 / max 0.87, 21% of pairs below 0.6, so 0.6 will reject some true returns (fails toward a new id). I have no different-person faces, so the false-merge side is NOT measured. Relatives' faces score high against each other, so wrong merges are most likely in families; the margin rule is the only defence and was tested only with synthetic look-alikes.
**Silent swaps:** `test_exchange_places_swaps_ids_silently_without_embeddings` shows that when people exchange places while occluded and there are no embeddings, ids follow the spots and `track_conf` stays high; write suspension cannot catch that (it only reacts to `track_conf`). With embeddings the ids stay correct. Synthetic embeddings (random orthogonal vectors, mixtures for look-alikes) show the logic works, not that real faces separate.
**Verified:** `tests_identity.py` (old node loads; node only after N consecutive frames and carries `visit_id`; broken streak and one-frame passer-by never become nodes; `end()` deletes nodes, edges and the gallery; writes dropped, counted and logged; end-to-end visit through the tracker in an EMPTY temp dir creates only the decision log, no `faces.npz`, `sessions.db`, `kg_state.json` or embedding file). 156 passed.
**Not verified:** real cameras or real faces through the new streak rule; behaviour in crowds.
**Note on the frozen simulator config:** adding `stable_frames` changed the effective-config hash; the first official run's hash (`1929eb86...`) belongs to commit e972257, not to this config.

---

## live demo: camera cut-outs / black panel  *(branch `museum-visual-profile`)*

**Report:** the feed sometimes cuts off or shows black.
**What I could and could not reproduce.** With the camera healthy the stream was solid: 150 s monitor, 2301 distinct frames, 15.3 fps, no gap over 1 s, no disconnect, no server error; and 6 rapid page reloads in Chrome all loaded and kept advancing (canvas-hash check). I could NOT reproduce the cut-off with a healthy camera.
**What I did find (two real weaknesses, one observed):**
- The capture thread never recovered if the camera stopped delivering: on a failed `read()` it just slept and retried forever on the same handle, so the panel froze or stayed black with no explanation. An unplug, a driver hiccup, or another program taking `/dev/video0` would look exactly like "the feed cut off". The camera being unavailable did actually happen once during testing (a fresh server could not open `/dev/video0` right after the previous one was killed), and it crashed the server at startup.
- The page only retried the `<img>` on an error event; a stream that silently stops advancing raised none.
**Fix:** `capture_loop` reopens the camera after 1.5 s without frames (exponential backoff to 3 s), publishes a placeholder picture saying why ("Camera stopped delivering frames / reconnecting", or the open error) instead of black, never exits after startup; start-up still fails fast but with a message to close other programs using the camera; a detection exception no longer kills the detection thread; `/state` has `camera: {status, fps, frame_age, reconnects, last_error}` shown under the video (red when unhealthy); the page has a watchdog that reloads the stream if the picture stops changing while the camera is healthy.
**Verified:** unit tests with a fake camera (open fails, delivers, stalls, reopens: status goes reconnecting -> ok, reconnects counted, something always published); live run after the change: status ok, 15 fps, 20 s monitor with no stalls. 150 tests pass.
**Not verified:** a real unplug/replug (I did not do one), and I do not know what caused the cut-offs you saw, so this fixes the unrecovered-stall path and makes the cause visible rather than proving it was the cause. If it recurs, the red status line under the video and `reconnects` in `/state` will say whether the camera or the browser stream dropped.

---

## live demo: lag, resizable UI, background-face fix  *(branch `museum-visual-profile`)*

**Problem 1, laggy feed.** Measured with a script that counts DISTINCT frames from `/video`: 4.2 fps idle, 3.6 fps during a chat turn with a 1.7 s freeze. Causes: the camera loop ran detection (MTCNN + embeddings + FairFace) inside the capture loop, so the picture was paced by the models; `chat()` held the state lock across both LLM calls (4-12 s), and `process_frame` needed that lock, so the camera froze mid-turn; the stream also resent duplicate frames.
**Fix:** `start_camera` runs a capture thread (buffer 1, no lock, no models, draws the overlay, publishes a JPEG) and a detection thread (newest frame only); `chat()` holds the state lock only for its quick state steps, never across the LLM calls (`chat_lock` serialises turns); `/video` sends only new frames.
**Verified:** 15.6 fps idle, 15.4 fps during a chat turn, longest gap 0.13 s (was 1.71 s). The camera itself delivers 15.3 fps raw (also with MJPG), so the stream is now at the hardware's limit. Test `test_slow_llm_does_not_block_frames_or_state`.
**Problem 2, resizable components.** Every card (camera, conversation, profile, graph, policy steps) has a native bottom-right drag handle (its content grows with it), the two columns have a drag splitter, and sizes are remembered per browser in localStorage (best effort, try/catch). Checked in Chrome by screenshot; the drag gesture itself I did not perform.
**Problem 3 (found while checking the UI), a second visitor from the background.** When the visitor looked down, the largest face in frame became the robot head on the shelf; the relative-size rule (>= 0.5 of the largest) kept it, and it got its own "child" visitor and greeting. Added an absolute floor, `--min-face-height` 0.18 of the frame height (lab scene measured: robot head <= 16%, visitor >= 19%), applied before the relative rule (`filter_faces`, unit-tested). Re-run: with nobody visible there are no visitors; when the visitor returned (116-129 px tall) exactly one.
**Limits:** the 0.18 floor is tuned to this lab scene and camera position and is tight (16% vs 19%); a different room needs recalibration, and a real visitor standing farther back than about 86 px face height is ignored.
**Not verified:** actually dragging the resize handles; more than one real person.

---

## live test with the real webcam and AI server  *(branch `museum-visual-profile`)*

**Setup:** webcam index 0 (640x480), lab AI server `10.42.0.118:7898` (now the default and in the README), live demo run with `--camera 0 --ai-server 10.42.0.118 --log-dir logs/museum`. Browser UI checked in Chrome.
**Worked:** camera frames, detection, FairFace on live frames (adult, confidence ~0.7-0.9 while the single seated visitor was in view), opening row `adult_solo`, AI-server chat (4-12 s per turn: one extraction call + one reply call, run synchronously), real LLM extraction produced a like ("dinosaurs") and a dislike ("loud crowded rooms") with no style inference from either, the explicit answer "The full story please" raised depth and took the first adaptation step, the opening hint was dropped as soon as real evidence existed, the UI redrew profile bars, graph edges and policy steps as it happened, and the decision log for the session contains no visitor text (4 lines, checked by grep).
**Didn't work -> fixed:**
- *Ghost visitors.* The first run created 3 visitors for 1 person. Measured on 53 real frames: Haar found a face in only 26/53 frames (it flickered), MTCNN in 53/53, but MTCNN always also found a 2nd face (a humanoid robot head in the background, 0.24-0.39 of the visitor's face area). Fixes: `--detector` (default now mtcnn), `--min-face-frac` 0.5 (drops faces smaller than half the largest), and a person is only created after `min_frames` confirmed frames, so a one-frame ghost can never become a visitor or inflate `group_size`. After the fix: 1 visitor, stable. Ceiling: a real second person standing much farther back is ignored too.
- *Black camera panel in Chrome.* The MJPEG `<img>` was in a broken state on first load (the server was fine: `curl` got real frames); reloading the image worked, so the page now retries on error. Root cause of the first failure not found.
**Observed, not fixed (design issues the live run exposed):**
- One explicit answer produced TWO adaptation steps (levels 1 then 2 by turn 3) with no new evidence for depth, because confidence stays at 0.8 >= 0.7 and each accepted step allows the next after the 2-turn cooldown. "One level at a time" holds, but the robot ramps to the cap on a single answer.
- The LLM invents exhibit content ("excavation site", "galleries") despite the "do not invent facts" instruction, because there is no exhibit catalogue to ground it (the README's open decision).
- Latency 4-12 s per turn is too slow for natural conversation; extraction could run after the reply.
**Not verified:** more than one person in front of the camera, a child, FairFace accuracy on real faces beyond "plausible", the misread/swap behaviour with real tracking, long visits.

---

## live demo: webcam + conversation + live KG view  *(branch `museum-visual-profile`)*

**Goal:** something runnable to meet a person, watch the profile and KG build live, and talk to the lab LLM.
**Built:** `modules/museum/live.py` (`Live` session + Flask app + camera thread + CLI), `live_ui.html` (video with debug overlay, chat, profile bars, SVG graph, policy-step table; polls `/state`), `modules/ai_server_llm.py` (vendored stdlib TCP client from ChatBox),
`ProfilingPolicy.note_external_question` (the opening's question uses question budget). The webcam path uses `FaceIdentifier` only for detection + embeddings (`identify_all(..., with_emb=True)`); it never enrols, saves or loads, and the ephemeral guard would raise if it did.
The age-band prior only picks the opening row and, while no real evidence exists, adds "use simple words" to the prompt; the LLM never sees the band, mood, or anything about appearance.
**Verified (headless):** `tests_live.py` (opening from the prior without age wording, unsure -> neutral row, answer -> evidence -> prior dropped, extraction -> likes + graph edges, one-level steps, negative reaction reverts + logs, low `track_conf` drops writes and the UI count shows it, end wipes and only the `.jsonl` log remains, disk `SessionStore` refused mid-visit, Flask endpoints, no-face case);
the server process also started and served `/chat`, `/state`, `/video` with the virtual visitor. 145 passed.
**Didn't work -> fixed:** graph label code evaluated `id.split(":")[1]` for person ids (IndexError, found by tests); two of my live tests mis-modelled the policy (one explicit answer already reaches the adapt gate, so the first step happens on that same turn, and the visitor's reaction is judged on the very next message).
**NOT verified:** a real webcam (this machine has no `/dev/video*`), real MTCNN/Haar detection + embeddings through `Detection`, real FairFace inference on live frames, the real AI server (unreachable from here), real LLM extraction quality and latency (extraction runs synchronously inside the chat request), multi-person visits in the UI (the chat addresses the largest face), the UI in a browser (only fetched, never rendered by me).

---

## RESULT: first official simulator run  *(branch `museum-visual-profile`)*

**Run:** 200 paired seeds (0..199), 24 turns, headline extraction noise 0.1 and age-misread 0.1, config hash `1929eb86...3d01d5` (matches the FROZEN entry), code at commit `e972257` (tree clean). Full tables: `docs/results/museum_first_run.md` / `.json`.
No threshold was changed before or after. Numbers are per visitor, mean [95% bootstrap CI over seeds]. The intervals reflect simulation randomness only, not uncertainty about real visitors or about my visitor model.
**What the simulator is:** my model of visitors (truthful 90% of the time, accept/decline rules, templated speech) and a fake LLM whose corruptions are mostly the kinds the guards drop. It is easy by construction; it says nothing about real people.

**Headline (six single-visitor scenarios):**
| | fixed | prior-only | profile |
|---|---|---|---|
| wrong openings | 0 | 0.25 [0.23, 0.27] | 0.25 [0.23, 0.27] |
| right adaptation steps | 0 | 0 | 3.17 [3.12, 3.21] |
| wrong adaptations, opposed to truth | 0 | 0 | 0.07 [0.05, 0.08] |
| wrong adaptations, made when truth was neutral | 0 | 0 | 0.00 [0.00, 0.00] |
| reversals | 0 | 0 | 0.37 [0.33, 0.40] |
| questions per visitor | 0 | 0 | 3.24 [3.20, 3.28] |
| declined offers | 0 | 0 | 0.07 [0.06, 0.09] |
| turns to confident | n/a | n/a | 4.78 [4.69, 4.88] |
| confident but wrong (of dims that got confident) | n/a | n/a | 0.08 [0.06, 0.09] |
| style sign correct / wrong / no call | n/a | n/a | 0.98 / 0.01 / 0.01 |

**Break-even (sensitivity analysis, not a measured cost):** `profile` stops beating `fixed` at wrong/right cost ratio W/R = 12.3 [11.4, 13.3] (q=0); 11.3 [10.5, 12.2] with a question cost of 0.1; 7.3 [7.0, 7.6] when the three swap scenarios are included.
`prior-only` stops beating `fixed` at W/R = 3.0 [2.7, 3.3]; at W/R = 5 its paired net benefit vs fixed is negative (-0.51 [-0.60, -0.41]). Misread sweep: prior-only break-even falls 5.0 -> 3.0 -> 2.0 -> 1.4 and profile 17.1 -> 12.3 -> 9.6 -> 7.7 for misread 0 / 0.1 / 0.2 / 0.3.

**Findings, with what supports them:**
1. *Silent swaps are the real risk, suspension helps.* Swap pair (quiet + chatty): detected (writes suspended, 2.8 dropped per visit) wrong-opposed 0.12, style-wrong 1%, MAE 0.15; silent (same swap, no `track_conf` drop) wrong-opposed 0.86, wrong-neutral 0.51, style-wrong 7%, MAE 0.47; silent-persistent style-correct 0.47 / wrong 0.44 (coin flip), confident-but-wrong 0.18. From the table means, persistent-silent profile has about 5.0 right vs 1.2 wrong adaptations per pair, i.e. `fixed` wins above W/R about 4 there (rough, from means, not bootstrapped).
2. *A wrong opening is never undone.* `profile` has exactly prior-only's wrong openings. The policy only changes later adaptation levels; nothing corrects the opening (vocabulary/gaze/rate) once real evidence arrives, and the simulator does not even model a lingering opening effect. This is a gap, not a result in favour of the design.
3. *One explicit answer is enough to adapt* (choice answer weight 0.8 >= the 0.7 adapt gate). So a visitor's single lapse (10% in the model) makes a wrong adaptation, which is then reverted: this plausibly explains the 8% confident-but-wrong and the 0.07 wrong steps. Not isolated by an experiment.
4. *Reversals mostly undo right steps (inference).* Reversals 0.37 vs wrong steps 0.07 per visitor; the model has visitors stay silent after 10% of right adaptations, and silence counts as "ignored" = negative. I did not count reversed-right steps directly; a "right steps wrongly reverted" metric should be added (that would be a new run).
5. *Neutral visitors cost questions without benefit:* topic_dislike (truth neutral on every style dimension) used the whole budget (4.00 questions) and made no adaptations (right or wrong). The wrong-neutral number is 0.00 everywhere except the swap scenarios.
6. *Extraction noise has a modest effect* (style-correct 0.99 -> 0.97, confident-but-wrong 0.07 -> 0.11 for noise 0 -> 0.3), partly by construction: 3 of my 4 corruption types are dropped by the real guards.
7. *like_precision is 0.63 and I first thought it was a bug.* Diagnostic (not an official run, no config change, seeds 0..59): precision is 1.00 at noise 0 and 0.63 at noise 0.1, so it is the injected spurious-topic noise (a real word of the utterance reported as a like, which the quote guard cannot catch), working as designed. The guards pass hallucinated topics that quote real words.
**Not verified:** anything about real visitors, real LLM extraction, real face re-association, or the hypotheses in findings 3-4.

---

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
