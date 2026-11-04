# modules/museum: low-regret personalisation for one-off visits

Steps A-E of the museum-visual-profile work. Evaluation (decision log, baseline modes, simulator) is **not built yet**.

| File | What |
|---|---|
| `appearance.py`, `worker.py` | Age band only (`child` / `adult` / `unsure`) per tracked face; smoothing, abstention, in-memory `visitor_N` tracker; `AgeBandWorker` thread |
| `initiation.py` | `choose_opening(band, confidence, group_size)` -> `OpeningParams` from `config/museum_initiation.yaml` |
| `profile.py`, `profile_extraction.py` | Visit-scoped likes / dislikes / style (`talk_level`, `depth`, `pace`) as `ProfileEdge`s in a private in-memory KG; guarded per-turn LLM extraction |
| `profiling_policy.py` | Question budget, confidence gates (generic / offer / adapt), one-step adaptation with reversal |
| `profile_api.py` | Read-only interface for the group policy |

## Rules (enforced by tests where noted)
- **Age band only.** No gender, race/ethnicity or personality is estimated, stored or returned. FairFace outputs them; the backend reads only the 9 age logits (test: output fields).
- **The prior is weak.** It only picks an opening row; never spoken, never evidence, never in `dimensions`; its weight halves with each real evidence item (tests).
- **Ephemeral.** Visit-scoped `visitor_N` ids; private `InMemoryGraphStore` that is never saved; `end_visit` deletes persons, edges, topics and style nodes; no files are created (test). Museum code must not call `FirstImpression.auto_enroll` (writes `faces.npz`) or open a disk `SessionStore`; **nothing enforces that beyond this code not importing them.**
- **No inferred mood** is extracted or put in any prompt.
- LLM proposes (JSON, closed vocab, confidence >= `CONFIDENCE_MIN`, verbatim quote); deterministic code decides what is written.

## Contract for the group policy
```python
from modules.museum.profile_api import start_visit, get_person_profile, get_visit_snapshot, end_visit
get_person_profile(visit_id, person_id) -> {
  "visit_id", "person_id",
  "dimensions": {"talk_level"|"depth"|"pace": {"value": -1..1, "confidence": 0..1,
                 "effective_value": value*confidence, "evidence_count": int, "low": str, "high": str}},
  "likes": [{"label","category","value","confidence","effective_value","evidence_count"}],
  "dislikes": [same],
  "prior": {"band","confidence","weight"} | None      # opening hint only; do NOT aggregate or treat as preference
}
get_visit_snapshot(visit_id) -> {"visit_id", "group_size", "persons": {person_id: <profile above>}}
```
Act on `effective_value`, not `value`: low-confidence items are already pulled toward neutral there. `value` -1 means the *low* end (quiet / brief / slow), +1 the *high* end (chatty / detailed / fast). Unknown visit or person raises `KeyError`. No group aggregation is implemented.

## Age model
FairFace ResNet-34 ONNX via onnxruntime (CPU), cached at `~/.fairface/fairface.onnx`, not downloaded automatically, SHA-256 pinned in `appearance.py`. Age bins are coarse, accuracy varies across groups, and this is a third-party ONNX conversion: weak prior only. The 10-19 bin counts toward neither child nor adult, so teenagers read `unsure`.

## Tests
`python3 -m pytest -q modules/graph_relationship/tests_*.py modules/museum/tests_*.py test_first_impression.py`
