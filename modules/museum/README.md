# modules/museum: low-regret personalisation for one-off visits

Steps A-F of the museum-visual-profile work.

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
python3 -m modules.museum.live --camera 0 --ai-server 172.24.30.251 --log-dir logs/museum
# no hardware, no network: virtual visitor, rule-based fake LLM
python3 -m modules.museum.live --no-camera --virtual-band adult --llm fake
```
Open http://127.0.0.1:8080. Face appears -> opening (greeting + one choice question) from the age-band prior -> each message: pending question/step outcome, guarded LLM extraction into the in-memory KG, policy decides the next move, reply uses the current style levels. The page redraws the profile, the graph and the policy steps every 0.7 s. "End visit (wipe)" deletes all person data. `--llm fake` only understands the simulator's exact phrases ("just give me the short version", "tell me the full story please", "let's keep things moving", "can we take this slowly", "I love X", "I can't stand X"); the opening depth question is also answered by keywords. Real webcam and real AI-server use are **not verified** (see PROGRESS.md).

## Identity (tracker) and suspended writes
`VisitTracker` keeps `visitor_N` ids by IoU plus, when the detector supplies `Detection(box, emb)`, face-embedding cosine (face outweighs position). A returning face takes back a lost id only if its best cosine is >= `reid_min_cos` (0.6) **and** beats the runner-up track by >= `reid_gap` (0.1); otherwise it is a new person (a wrongly new id loses a profile, a wrong merge corrupts one). Each track has a `track_conf`; an ambiguous match (small margin, or overlapping faces with no embeddings) drops it. `VisitProfile.apply_tracks(results)` passes it in; below `min_track_conf` (0.5) evidence writes are **dropped, not buffered**, and counted (`suspended_writes`, `suspended_by_person`); the policy asks nothing of that person.
- Face embeddings are biometric data held **in memory only**, per track, discarded with the track and by `AgeBandPipeline.clear()`; never written to disk.
- **Silent swaps remain possible**: if people exchange places while occluded and there are no embeddings, ids follow the spots and `track_conf` does not drop (a test shows this).
- The 0.6 / 0.1 values are not validated on real different-person faces; see PROGRESS.md (follow-up 5).

## Age model
FairFace ResNet-34 ONNX via onnxruntime (CPU), cached at `~/.fairface/fairface.onnx`, not downloaded automatically, SHA-256 pinned in `appearance.py`. Age bins are coarse, accuracy varies across groups, and this is a third-party ONNX conversion: weak prior only. The 10-19 bin counts toward neither child nor adult, so teenagers read `unsure`.

## Tests
`python3 -m pytest -q modules/graph_relationship/tests_*.py modules/museum/tests_*.py test_first_impression.py`
