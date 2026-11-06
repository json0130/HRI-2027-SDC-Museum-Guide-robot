# modules/museum: low-regret personalisation for one-off visits

## Scope and status (rescoped)

The visit flow has six steps. Only 1-4 are in scope and settled enough to build on:

| # | Step | Status |
|---|---|---|
| 1 | Check the visitor's face, keep one id for the whole visit | in scope, built (tracker, re-association, `track_conf`) |
| 2 | The person is a visit-scoped node in the KG | in scope, built (node after N stable frames, `visit_id`, wiped by `end()`) |
| 3 | Visual first impression: age band **plus coarse group context** | built (`group.py`): `group_id`, `group_size`, `has_child_band_member` only, never a relationship label; placeholders |
| 4 | Start the first interaction when the person is close enough | built (`engagement.py`): face-box ratio trigger, one event per group; placeholders |
| 5 | Greeting wording and the first choice question | **RESEARCH FIRST. Provisional placeholder.** |
| 6 | Collecting information and building the profile | **RESEARCH FIRST. Provisional placeholder.** |

**Provisional placeholders pending literature review (frozen: bug fixes only, no new question types, templates, dimensions, thresholds or policy logic):**
`initiation.py` + `config/museum_initiation.yaml` (greeting and first question wording, rows), `profile.py` / `profile_extraction.py` / `config/museum_profile.yaml`
(dimensions `talk_level`/`depth`/`pace`, evidence weights), `profiling_policy.py` / `config/museum_policy.yaml` (gates, budget, steps, question wording), and the
conversation half of `live.py` (keyword answer matching, style prompt). They are untested guesses: **nothing here shows that the greeting, the first question or the
profiling policy works.** Open questions are in `docs/RESEARCH_QUESTIONS_museum.md`.

`simulate.py` and `decision_log.py` were built before this rescope. The simulator's profile modes (`fixed` / `prior-only` / `profile`), its first run and the break-even
numbers evaluate the step-5/6 placeholders against my own model of visitors; they are not evidence about real visitors or about whether personalisation helps. They are
frozen and will not be extended. The decision log is kept (it is the one file allowed to outlive a visit) and will be reused for steps 1-4.

Steps A-F of the museum-visual-profile work (original numbering):

| File | What |
|---|---|
| `appearance.py`, `worker.py` | Age band only (`child` / `adult` / `unsure`) per tracked face; smoothing, abstention, in-memory `visitor_N` tracker; `AgeBandWorker` thread |
| `initiation.py` | `choose_opening(band, confidence, group_size)` -> `OpeningParams` from `config/museum_initiation.yaml` |
| `profile.py`, `profile_extraction.py` | Visit-scoped likes / dislikes / style (`talk_level`, `depth`, `pace`) as `ProfileEdge`s in a private in-memory KG; guarded per-turn LLM extraction |
| `profiling_policy.py` | Question budget, confidence gates (generic / offer / adapt), one-step adaptation with reversal |
| `profile_api.py` | Read-only interface for the group policy |
| `decision_log.py` | Anonymised JSONL decision log, the only thing kept after a visit (random `visit_id`, `t` = seconds since visit start, no text/labels/names/images/embeddings) |
| `simulate.py` | No-hardware simulator: `fixed` / `prior-only` / `profile` modes, scripted visitors with ground truth, paired seeds, bootstrap intervals, sweeps, break-even sensitivity analysis. `python -m modules.museum.simulate --seeds 200 --out docs/results --prefix museum_first_run` |

## Rules (enforced by tests where noted)
- **Age band only.** No gender, race/ethnicity or personality is estimated, stored or returned. FairFace outputs them; the backend reads only the 9 age logits (test: output fields).
- **The prior is weak.** It only picks an opening row; never spoken, never evidence, never in `dimensions`, and dropped entirely once any real evidence exists (tests). `get_visit_snapshot` / `get_person_profile` omit it unless `include_prior=True`.
- **Ephemeral.** Visit-scoped `visitor_N` ids; private `InMemoryGraphStore` that is never saved; `end_visit` deletes persons, edges, topics and style nodes; no files are created (test). While a `VisitProfile` is alive, `modules/ephemeral.py` makes a disk `SessionStore`, `FaceIdentifier.save/load`, `FirstImpression.auto_enroll/learn_name` and `GraphStore.save` raise `EphemeralViolation` (tests). It guards those entry points only; other file writes are not caught.
- **No inferred mood** is extracted or put in any prompt.
- LLM proposes (JSON, closed vocab, confidence >= `CONFIDENCE_MIN`, verbatim quote); deterministic code decides what is written.

## Contract for the group policy
```python
from modules.museum.profile_api import start_visit, get_person_profile, get_visit_snapshot, end_visit
get_person_profile(visit_id, person_id, include_prior=False) -> {
  "visit_id", "person_id",
  "dimensions": {"talk_level"|"depth"|"pace": {"value": -1..1, "confidence": 0..1,
                 "effective_value": value*confidence, "evidence_count": int, "low": str, "high": str}},
  "likes": [{"label","category","value","confidence","effective_value","evidence_count"}],
  "dislikes": [same],
  # only with include_prior=True:
  "prior": {"band","confidence","use": "opening_only"} | None   # None once any real evidence exists; never a preference
}
get_visit_snapshot(visit_id, include_prior=False) -> {"visit_id", "group_size", "persons": {person_id: <profile above>}}
```
Act on `effective_value`, not `value`: low-confidence items are already pulled toward neutral there. `value` -1 means the *low* end (quiet / brief / slow), +1 the *high* end (chatty / detailed / fast). Unknown visit or person raises `KeyError`. No group aggregation is implemented.

## Live demo (webcam + conversation + live KG view)
```bash
# real webcam + lab AI server (needs torch/facenet-pytorch/opencv/flask + the cached FairFace model)
python3 -m modules.museum.live --camera 0 --ai-server 10.42.0.118 --log-dir logs/museum
# no hardware, no network: virtual visitor, rule-based fake LLM
python3 -m modules.museum.live --no-camera --virtual-band adult --llm fake
```
Open http://127.0.0.1:8080. Face appears -> opening (greeting + one choice question) from the age-band prior -> each message: pending question/step outcome, guarded LLM extraction into the in-memory KG, policy decides the next move, reply uses the current style levels. The page redraws the profile, the graph and the policy steps every 0.7 s. "End visit (wipe)" deletes all person data. `--llm fake` only understands the simulator's exact phrases ("just give me the short version", "tell me the full story please", "let's keep things moving", "can we take this slowly", "I love X", "I can't stand X"); the opening depth question is also answered by keywords. Real webcam and real AI-server use are **not verified** (see PROGRESS.md).

## Engagement trigger (step 4)
`engagement.py` / `config/museum_engagement.yaml` emits an `EngageEvent(person_ids, group_id, distance_proxy, t, group, reason)`; the caller then runs `opening_for_event(event, profile)` (prior + group context as they are at that moment). Robot-agnostic: a Pepper adapter consumes the event later.
- **Distance proxy = face-box height / frame height. It is not a distance.** `enter_ratio` 0.20, `exit_ratio` 0.15 (hysteresis: between them the count holds, below exit it resets), `dwell_frames` 20, `min_track_conf` 0.6, `cooldown_s` 30, `max_attempts_per_person` 1. **All placeholders.** Measure your site with `python3 tools/calibrate_engagement.py --camera 0 --distances 0.5 1.0 2.0 --out docs/results/engagement_calibration.json` (stand at each tape-measured distance when asked; it stores only distances and ratios and prints a suggestion without editing any config).
- Fires only for a stable track (node exists) above `min_track_conf` that stayed beyond `enter_ratio` for `dwell_frames`; passers-by do not reach it. When the first member passes, it waits up to `group_settle_s` (4 s) for the group context: known -> one event for the whole group at once; still unknown at the deadline -> fires with the context **unknown** and the plain neutral opening.
- After an event, a person who is absent for `leave_s` (3 s) before the exchange starts is **disengaged** for the rest of the visit and never re-greeted; a return is logged, not greeted. (If the tracker gives them a new id on return, the engine cannot know: a limit.)
- **Known limitation, not corrected:** a child's face is smaller, so at the same real distance the ratio is lower and the trigger fires later (or never) for children. It is deliberately not compensated with the age band. Tested with scaled boxes. Pepper's depth sensing should replace the proxy.

## Coarse group context (step 3)
`group.py` / `config/museum_group.yaml`: output is only `group_id`, `group_size` and `has_child_band_member` (true / false / unknown). It comes from proximity (edge gap <= 1.5 x the larger face width) AND co-movement (arrived within 4 s, same heading or both still) held for 3 s of the last 10 s. **No relationship labels are inferred** (couple, family, parent/child, friends): that would be appearance-based guessing of social facts. Weak, opening-only, never spoken, never a KG fact, dropped once real evidence exists, hidden from `get_visit_snapshot` unless `include_group=True`. Any instability (membership change, merge of two groups, more than 6 people) reports **unknown**, and `choose_opening(..., group=ctx)` then uses the plain neutral opening. All thresholds are untuned placeholders; image distance is not real distance and side-by-side queues look like groups.

## Identity (tracker) and suspended writes
`VisitTracker` keeps `visitor_N` ids by IoU plus, when the detector supplies `Detection(box, emb)`, face-embedding cosine (face outweighs position). A returning face takes back a lost id only if its best cosine is >= `reid_min_cos` (0.6) **and** beats the runner-up track by >= `reid_gap` (0.1); otherwise it is a new person (a wrongly new id loses a profile, a wrong merge corrupts one). Each track has a `track_conf`; an ambiguous match (small margin, or overlapping faces with no embeddings) drops it. `VisitProfile.apply_tracks(results)` passes it in; below `min_track_conf` (0.5) evidence writes are **dropped, not buffered**, and counted (`suspended_writes`, `suspended_by_person`); the policy asks nothing of that person.
- Face embeddings are biometric data held **in memory only**, per track, discarded with the track and by `AgeBandPipeline.clear()`; never written to disk.
- **Silent swaps remain possible**: if people exchange places while occluded and there are no embeddings, ids follow the spots and `track_conf` does not drop (a test shows this).
- The 0.6 / 0.1 values are not validated on real different-person faces; see PROGRESS.md (follow-up 5).

## Age model
FairFace ResNet-34 ONNX via onnxruntime (CPU), cached at `~/.fairface/fairface.onnx`, not downloaded automatically, SHA-256 pinned in `appearance.py`. Age bins are coarse, accuracy varies across groups, and this is a third-party ONNX conversion: weak prior only. The 10-19 bin counts toward neither child nor adult, so teenagers read `unsure`.

## Tests
`python3 -m pytest -q modules/graph_relationship/tests_*.py modules/museum/tests_*.py test_first_impression.py`
