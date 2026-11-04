# HRI 2027 SDC — Museum Guide Robot

Low-regret personalization for one-off group visits.

A museum guide robot serves groups who meet it once, for an hour or two: singles, couples and
families, who want different things. Some prefer quiet listening, others prefer questions and
conversation. Personalizing normally needs lots of data and time, which a one-off visit doesn't
give, and a wrong guess can leave visitors worse off than a generic tour.

## Approach

The robot starts every visit with a good generic group tour. It then learns a few interaction-style
preferences (**talk level, depth, pace, proactivity**) from two sources:

- a cheap explicit question, and
- behaviour cues: dwell time, distance, and the questions the group asks.

Each preference carries a **confidence** value, and confidence decides what the robot does:

| Confidence | Behaviour |
|---|---|
| Low | Stay generic |
| Medium | Offer a choice ("Want more detail on this one?") |
| High | Adapt, in a way that is easy to undo |

A coarse age band (`child` / `adult` / `unsure`) from the camera only shapes how the robot *opens*; it is never spoken, never stored as a preference and is overwritten by what visitors say or choose. It never estimates gender, ethnicity or personality, and it forgets everything when the visit ends.

## Research questions

1. How much can a robot reliably learn about a group in the first few minutes?
2. Which adaptations cost little when the guess is wrong?
3. How should it handle a group whose members want different things?

**Proposed comparison:** generic (fixed mode) vs. context-only vs. low-regret adaptive, with the
accuracy of the robot's guesses deliberately varied.

## Status

Early. So far this repo contains only the first-impression code imported from
[ChatBox-companion-robot](../ChatBox-companion-robot) (copied from commit `65d36f3`). The
estimates-with-confidence, decision step, engagement cues and explicit ask are not built yet.
Face counting, speech input, noise sensing, hardware drivers, the session log, the style table
and the dialogue state machine live in the sibling `Library-Book-Robot` repo and have not been
reconciled with this one.

### What is here

| Path | What it is |
|---|---|
| `modules/first_impression.py` | Meet-a-stranger logic: `extract_name`, `slug_name`, and `FirstImpression` (`next_guest_id`, `auto_enroll`, `learn_name`, `meet`) |
| `modules/kg_extraction.py`, `modules/graph_relationship/extraction.py` | Guards: the LLM proposes, deterministic checks (JSON parse, enum, hallucination, confidence, normalize) decide what is kept |
| `modules/graph_relationship/` | Minimal graph closure the guards and rename depend on (`schema`, `store`, `interactions`, `topics`, `scales`, `rename`) and its `tests_*.py` |
| `modules/session_store.py` | Transcript store |
| `modules/face_webcam/face_id.py` | `FaceIdentifier` (enroll / identify_all / rename / save) |
| `modules/museum/` | Museum mode (see its [README](modules/museum/README.md)): age-band prior, opening table, visit-scoped profile in the KG, profiling policy, interface for the group policy. Evaluation not built yet |
| `config/museum_*.yaml` | Tunable thresholds, weights, wording |
| `test_first_impression.py` | Headless check of the first-impression flow |
| `docs/PROGRESS.md` | Provenance, what was verified, and open decisions |

## Running the checks

No camera, LLM or network needed.

```bash
python3 test_first_impression.py
python3 -m pytest -q modules/graph_relationship/tests_*.py modules/museum/tests_*.py
```

Python dependencies: `pydantic`, `numpy`, `pytest`, `pyyaml` (museum mode); `onnxruntime` + `opencv-python` only for the real FairFace backend. Importing `face_id.py` also needs `opencv-python`,
`torch` and `Pillow`, and real face recognition needs `facenet_pytorch`.

## Open decisions

- **Persistence vs. forgetting.** The imported auto-enrol and name learning save faces, names and
  the graph to disk. The design here forgets everything when the visit ends and uses anonymous
  in-memory IDs, so this needs an in-memory / wipe-at-end mode, or only the guards and anonymous IDs
  should be kept.
- **Reconciling with Library-Book-Robot.** Decide what is reused from there instead of imported here.
- **Catalogue grounding.** The LLM is to be grounded in a catalogue (books in the Library-Book-Robot
  repo; exhibits later).

See [docs/PROGRESS.md](docs/PROGRESS.md) for details.

## License

See [LICENSE](LICENSE).
