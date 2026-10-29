# Progress log

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
