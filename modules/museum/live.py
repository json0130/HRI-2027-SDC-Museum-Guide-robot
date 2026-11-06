"""Live museum demo: webcam + conversation (LLM on the lab AI server) + live visit-scoped KG view.

    python -m modules.museum.live --camera 0 --ai-server HOST[:PORT]              # real webcam + real LLM
    python -m modules.museum.live --no-camera --virtual-band adult --llm fake     # no hardware, no network

Open http://127.0.0.1:8080 . Flow: a face appears -> the age-band prior picks only the OPENING (greeting +
one choice question, never mentioning age/looks) -> every visitor message is (1) mapped to an answer/outcome for
the pending question or step, (2) run through the guarded LLM extraction into the in-memory KG, (3) the profiling
policy decides the next move (ask / offer / adapt one level / nothing), (4) the reply is generated with the
CURRENT style levels. The page polls /state and redraws the profile and the graph. Nothing is written to disk
except the anonymised decision log (--log-dir); frames, embeddings and chat text stay in memory and are wiped
by "End visit". Mood is never inferred or put in a prompt; the prior never reaches the LLM except as "use simple
words" for the opening vocabulary, and only until real evidence exists.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

import numpy as np

from . import profile_api
from .appearance import AgeBandPipeline, AgeProbs, Detection, FakeAgeEstimator, FairFaceAgeEstimator
from .decision_log import DecisionLog, snapshot
from .initiation import OpeningParams, choose_opening
from .profile import DIMS
from .profile_extraction import build_system_prompt, extract_and_apply_profile
from .profiling_policy import ProfilingPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
CHOICE_WORDS = {  # dim -> (low-end words, high-end words); only used for the dimension just asked
    "depth": ({"short", "brief", "quick", "summary", "shorter"}, {"full", "story", "detail", "detailed", "everything", "more"}),
    "talk_level": ({"quiet", "silent", "myself", "alone"}, {"chat", "talk", "tell", "explain", "chatty"}),
    "pace": ({"slow", "slowly", "relaxed", "leisurely"}, {"moving", "fast", "faster", "hurry", "brisk", "quicker"}),
}
YES = {"yes", "yeah", "yep", "sure", "please", "ok", "okay", "definitely", "go"}
NO = {"no", "nope", "not", "don't", "dont", "stop", "skip", "later", "nah", "enough"}


def _words(text: str) -> set:
    return set("".join(c if c.isalnum() or c == "'" else " " for c in text.lower()).split())


def interpret_choice(dim: str, text: str) -> int:
    lo, hi = CHOICE_WORDS[dim]
    w = _words(text)
    a, b = len(w & lo), len(w & hi)
    return -1 if a > b else (1 if b > a else 0)


def style_instructions(levels: Dict[str, int], simple_words: bool) -> str:
    d, t, p = levels["depth"], levels["talk_level"], levels["pace"]
    out = ["Brief and warm; you are a museum guide robot talking with one visitor."]
    out.append("Keep it to ONE short sentence." if d < 0 else ("Give richer detail and background (4-6 sentences)." if d > 0 else "Use 2-3 sentences."))
    out.append("Be minimal: no small talk and no follow-up questions." if t < 0 else ("Be conversational and friendly." if t > 0 else "Be natural."))
    out.append("Be slow and calm in phrasing." if p < 0 else ("Be brisk and get to the point." if p > 0 else ""))
    if simple_words:
        out.append("Use simple, everyday words.")
    out.append("Do not invent facts about specific exhibits you cannot see. Never mention the visitor's age or appearance.")
    return " ".join(x for x in out if x)


class Live:
    """All state for one live museum session. Thread-safe via one lock; camera thread calls process_frame."""

    def __init__(self, llm, extract_llm: Callable[[str, str], str], estimator, detect_fn, *, log_dir: Optional[str] = None,
                 reply_fn: Optional[Callable] = None):
        self.llm, self.extract_llm, self.reply_fn = llm, extract_llm, reply_fn
        self._estimator, self._detect, self._log_dir = estimator, detect_fn, log_dir
        self.lock = threading.RLock()          # state; held only briefly
        self.chat_lock = threading.Lock()      # one conversation turn at a time
        self.pipe_lock = threading.Lock()      # the tracker/smoothers are touched by one detection pass at a time
        self.jpeg: Optional[bytes] = None
        self.jpeg_id = 0
        self.frame = None                      # latest raw frame (capture thread -> detection thread)
        self.frame_id = 0
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.visit_id = uuid.uuid4().hex[:8]
            self.profile = profile_api.start_visit(self.visit_id)
            self.policy = ProfilingPolicy(self.profile)
            with self.pipe_lock:
                self.pipeline = AgeBandPipeline(self._estimator, self._detect)
            self.log = DecisionLog(self._log_dir, "profile") if self._log_dir else None
            self.met: Dict[str, OpeningParams] = {}
            self.prior_log: Dict[str, dict] = {}
            self.faces: List[dict] = []
            self.messages: List[dict] = []
            self.history: Dict[str, list] = {}
            self.turn = 0
            self.opening_dim: Dict[str, Optional[str]] = {}
            self.last_extraction: dict = {}
            self.ended = False

    # -- camera side ----------------------------------------------------------

    def process_frame(self, frame) -> List[dict]:
        with self.pipe_lock:
            if self.ended:
                return []
            res = self.pipeline.process(frame)           # heavy (detection + embeddings + age): no state lock held
        with self.lock:
            if self.ended:
                return []
            min_frames = self.pipeline.cfg["min_frames"]
            confirmed = [r for r in res if r["n_frames"] >= min_frames]   # a one-frame ghost never becomes a visitor
            for r in confirmed:
                if r["person_id"] not in self.profile.persons:
                    self.profile.add_person(r["person_id"])
            self.profile.apply_tracks(res)
            for r in confirmed:
                if r["person_id"] not in self.met:
                    self._meet(r, len(confirmed))
            self.faces = res
            return res

    def _meet(self, r: dict, group_size: int) -> None:
        pid = r["person_id"]
        op = choose_opening(r["band"], r["confidence"], group_size)
        self.met[pid] = op
        self.profile.set_prior(pid, r["band"], r["confidence"])
        self.prior_log[pid] = {"band": r["band"], "confidence": r["confidence"]}
        self.opening_dim[pid] = "depth" if "short version" in op.first_question else None
        self.policy.note_external_question(pid, self.turn)
        self.messages.append({"who": "robot", "pid": pid, "text": f"{op.greeting('your museum guide')} {op.first_question}",
                              "meta": f"opening row {op.row_id}"})

    def primary(self) -> Optional[str]:
        best, area = None, -1
        for f in self.faces:
            b = f["box"]
            a = (b[2] - b[0]) * (b[3] - b[1])
            if f["person_id"] in self.met and a > area:
                best, area = f["person_id"], a
        return best or next(iter(self.met), None)

    # -- conversation ---------------------------------------------------------

    def chat(self, text: str) -> dict:
        """One conversation turn. `chat_lock` serialises turns; the state lock is held only for the quick state
        steps, never across the two LLM calls (4-12 s), so the camera, detection and /state keep running."""
        with self.chat_lock:
            with self.lock:
                text = (text or "").strip()
                pid = self.primary()
                if self.ended or not text:
                    return {"error": "visit ended" if self.ended else "empty message"}
                self.messages.append({"who": "visitor", "pid": pid or "?", "text": text})
                if pid is None:
                    self.messages.append({"who": "robot", "pid": "-", "text": "I can't see you yet. Please look at the camera."})
                    return {"reply": self.messages[-1]["text"]}
                self.turn += 1
                t, st = self.turn, self.policy._st(pid)
                pend, outcome = st.pending, "none"
                # (1) outcome of the pending question / offer / adaptation
                if pend and pend["kind"] == "ask_choice":
                    s_ = interpret_choice(pend["dim"], text)
                    if s_:
                        self.policy.answer_choice(pid, pend["dim"], s_, t); outcome = "answered"
                elif pend and pend["kind"] == "offer":
                    w = _words(text)
                    if w & NO and not w & YES: outcome = "declined"
                    elif w & YES: outcome = "accepted"
                    if outcome != "none": self.policy.observe(pid, outcome, t)
                elif pend and pend["kind"] == "adapt":
                    outcome = "declined" if (_words(text) & NO) else "accepted"
                    self.policy.observe(pid, outcome, t)
                elif self.opening_dim.get(pid) and pend is None:
                    s_ = interpret_choice(self.opening_dim[pid], text)
                    if s_:
                        self.policy.answer_choice(pid, self.opening_dim[pid], s_, t); outcome = "answered"
                    self.opening_dim[pid] = None
            # (2) guarded extraction: LLM call WITHOUT the state lock, then apply under it
            try:
                raw = self.extract_llm(build_system_prompt(), text)
            except Exception as exc:  # noqa: BLE001 - an LLM outage must not break the visit
                raw, err = None, exc
            def _llm(system, user):
                if raw is None:
                    raise err
                return raw
            with self.lock:
                if self.ended:
                    return {"error": "visit ended"}
                self.last_extraction = extract_and_apply_profile(self.profile, pid, text, _llm, t)
                # (3) next move
                move = self.policy.next_move(pid, t)
                levels = self.policy.levels(pid)
                op = self.met[pid]
                simple = op.vocabulary == "simple" and self.profile.prior(pid) is not None   # opening-only; gone after evidence
                system = style_instructions(levels, simple)
                hist = list(self.history.setdefault(pid, []))
                extraction = self.last_extraction
            # (4) reply with the CURRENT style levels (LLM call without the lock)
            if self.reply_fn:
                reply = self.reply_fn(system, text)
            else:
                reply = self.llm.respond(system, text, history=hist, max_tokens=60 if levels["depth"] < 0 else (200 if levels["depth"] > 0 else 120))
            if not reply or reply.startswith("[LLM"):
                reply = "(the language model is not available right now)"
            with self.lock:
                if self.ended:
                    return {"error": "visit ended"}
                self.history.setdefault(pid, []).append((text, reply))
                full = reply + (("\n" + move.text) if move.text else "")
                self.messages.append({"who": "robot", "pid": pid, "text": full,
                                      "meta": f"move={move.kind}" + (f" {move.dim}" if move.dim else "") + (f" level→{move.level}" if move.level is not None else "")})
                if self.log:
                    self.log.log(pid, prior=self.prior_log.get(pid), profile=snapshot(self.profile, pid),
                                 action={"kind": move.kind, "dim": move.dim, "direction": move.direction, "level": move.level},
                                 gate=self.policy.gate(self.profile.style(pid, move.dim)["confidence"]) if move.dim else "none",
                                 outcome=outcome, reversal=any(r["reversal"] and r["turn"] == t for r in self.policy.step_log),
                                 track_conf=self.profile._track_conf.get(pid, 1.0), suspended_writes=self.profile.suspended_writes)
                return {"reply": full, "move": move.kind, "outcome": outcome, "extraction": extraction}

    def end(self) -> dict:
        with self.lock:
            if self.ended:
                return {"ended": True}
            summary = profile_api.end_visit(self.visit_id)
            with self.pipe_lock:
                self.pipeline.clear()
            if self.log: self.log.close()
            self.messages, self.history, self.met, self.faces = [], {}, {}, []
            self.ended = True
            return summary

    # -- view -----------------------------------------------------------------

    def state(self) -> dict:
        with self.lock:
            if self.ended:
                return {"visit_id": self.visit_id, "ended": True, "messages": [], "faces": [], "persons": {}, "graph": {"nodes": [], "edges": []}}
            snap = profile_api.get_visit_snapshot(self.visit_id, include_prior=True)
            nodes, edges = [], []
            for n in self.profile.store._nodes.values():
                if n.node_type in ("person", "topic", "style_dim"):
                    nodes.append({"id": n.id, "type": n.node_type,
                                  "label": n.dim if n.node_type == "style_dim" else (n.id if n.node_type == "person" else n.label)})
            for e in self.profile.store._edges.values():
                if e.edge_type == "visit_pref":
                    edges.append({"source": e.source_id, "target": e.target_id, "value": e.value, "confidence": e.confidence,
                                  "evidence": len(e.evidence), "sources": sorted({x.source for x in e.evidence})})
            return {"visit_id": self.visit_id, "ended": False, "faces": [{k: v for k, v in f.items() if k != "box"} | {"box": list(f["box"])} for f in self.faces],
                    "persons": snap["persons"], "group_size": snap["group_size"],
                    "levels": {p: self.policy.levels(p) for p in self.profile.persons},
                    "pending": {p: self.policy._st(p).pending for p in self.profile.persons},
                    "met": {p: o.to_dict() for p, o in self.met.items()},
                    "step_log": self.policy.step_log[-14:], "suspended_writes": self.profile.suspended_writes,
                    "last_extraction": self.last_extraction, "messages": self.messages[-60:], "graph": {"nodes": nodes, "edges": edges},
                    "thresholds": self.policy.cfg["thresholds"]}


# ── web layer ────────────────────────────────────────────────────────────────────────────

def make_app(live: Live):
    from flask import Flask, Response, jsonify, request
    app = Flask(__name__)
    html = open(os.path.join(HERE, "live_ui.html"), encoding="utf-8").read()

    @app.get("/")
    def index(): return Response(html, mimetype="text/html")

    @app.get("/state")
    def state(): return jsonify(live.state())

    @app.post("/chat")
    def chat(): return jsonify(live.chat((request.get_json(silent=True) or {}).get("text", "")))

    @app.post("/end")
    def end(): return jsonify(live.end())

    @app.post("/new")
    def new():
        live.end(); live.reset(); return jsonify({"visit_id": live.visit_id})

    @app.get("/video")
    def video():
        def gen():
            last = -1
            while True:
                if live.jpeg_id != last and live.jpeg:
                    last = live.jpeg_id
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + live.jpeg + b"\r\n"
                else:
                    time.sleep(0.01)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    return app


def start_camera(live: Live, source, stop: Optional[threading.Event] = None) -> List[threading.Thread]:
    """Two threads so the picture never waits for the models:
    capture: read frames as fast as the camera delivers (buffer of 1, so no stale backlog), draw the latest face
             overlay (debug only) and publish a JPEG. It takes no lock and does no model work.
    detect : run face detection + embeddings + age on the NEWEST frame, as fast as it can (older frames are skipped).
    `source` is a cv2 index, or None for a virtual camera."""
    import cv2
    stop = stop or threading.Event()
    cap = None
    if source is not None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera {source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def capture():
        while not stop.is_set():
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02); continue
            else:
                frame = np.full((300, 400, 3), 40, np.uint8); time.sleep(0.04)
            live.frame, live.frame_id = frame, live.frame_id + 1
            img = frame.copy()
            for f in live.faces:                      # reference read; replaced atomically by the detect thread
                x1, y1, x2, y2 = f["box"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 120), 2)
                cv2.putText(img, f"{f['person_id']} {f['band']} tc={f['track_conf']:.1f}", (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 1)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                live.jpeg, live.jpeg_id = buf.tobytes(), live.jpeg_id + 1

    def detect():
        seen = -1
        while not stop.is_set():
            if live.frame is None or live.frame_id == seen:
                time.sleep(0.005); continue
            seen, frame = live.frame_id, live.frame
            live.process_frame(frame)

    threads = [threading.Thread(target=capture, daemon=True), threading.Thread(target=detect, daemon=True)]
    for t in threads:
        t.start()
    return threads


def filter_faces(dets: List[Detection], frame_h: int, min_frac: float, min_height_frac: float) -> List[Detection]:
    """Drop background faces (posters, robot heads). Two rules, both site-calibration knobs:
    absolute: face height must be >= min_height_frac of the frame height (so a small background face is never
              promoted just because the visitor looked down and left the frame);
    relative: face area must be >= min_frac of the largest remaining face.
    Ceiling: a real second person standing far back is ignored too."""
    big_enough = [d for d in dets if (d.box[3] - d.box[1]) >= min_height_frac * frame_h]
    area = lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
    big = max((area(d) for d in big_enough), default=0)
    return [d for d in big_enough if area(d) >= min_frac * big]


def build(args) -> tuple:
    if args.llm == "fake":
        from .simulate import make_fake_llm
        extract_llm = make_fake_llm(0.0, ("live",))
        reply_fn = lambda system, text: "Interesting! " + ("Let me tell you more about that." if "richer" in system else "Noted.")
        llm = None
    else:
        from modules.ai_server_llm import from_arg
        llm = from_arg(args.ai_server)
        if not llm.available:
            raise SystemExit("AI server not reachable; check --ai-server HOST[:PORT] (or use --llm fake)")
        extract_llm = lambda system, user: llm.respond(system, user, json_mode=True, max_tokens=300, temperature=0.0)
        reply_fn = None
    if args.no_camera:
        band = args.virtual_band
        probs = {"child": AgeProbs(0.9, 0.08, 0.02), "adult": AgeProbs(0.06, 0.92, 0.02), "unsure": AgeProbs(0.4, 0.4, 0.2)}[band]
        estimator = FakeAgeEstimator({"visitor_1": probs})
        emb = np.random.RandomState(7).randn(512)
        detect = lambda f: [Detection((100, 60, 240, 220), emb / np.linalg.norm(emb))]
    else:
        from modules.face_webcam.face_id import FaceIdentifier   # heavy import (torch); detection + embeddings only,
        fid = FaceIdentifier(detector=args.detector)             # NO enroll / save / load: museum mode forbids them
        estimator = FairFaceAgeEstimator() if args.age_backend == "fairface" else FakeAgeEstimator({}, 0)

        def detect(f):
            ds = [Detection(tuple(int(v) for v in r[2]), r[3]) for r in fid.identify_all(f, max_faces=4, with_emb=True)]
            return filter_faces(ds, f.shape[0], args.min_face_frac, args.min_face_height)
    return Live(llm, extract_llm, estimator, detect, log_dir=args.log_dir, reply_fn=reply_fn)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live museum demo (webcam + chat + live KG view)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-camera", action="store_true", help="virtual visitor, no hardware")
    ap.add_argument("--virtual-band", choices=["child", "adult", "unsure"], default="adult")
    ap.add_argument("--ai-server", default=os.environ.get("AI_HOST", "10.42.0.118"), help="HOST[:PORT] of the lab AI server")
    ap.add_argument("--llm", choices=["ai-server", "fake"], default="ai-server")
    ap.add_argument("--age-backend", choices=["fairface"], default="fairface")
    ap.add_argument("--detector", choices=["opencv", "mtcnn"], default="mtcnn",
                    help="mtcnn detected the seated visitor in 53/53 test frames, Haar in 26/53")
    ap.add_argument("--min-face-frac", type=float, default=0.5, help="ignore faces smaller than this fraction of the largest face's area")
    ap.add_argument("--min-face-height", type=float, default=0.18,
                    help="ignore faces shorter than this fraction of the frame height (lab scene: robot head <=16%%, visitor >=19%%)")
    ap.add_argument("--log-dir", default=None, help="write the anonymised JSONL decision log here")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    if a.llm == "ai-server" and not a.ai_server:
        ap.error("--ai-server HOST[:PORT] is required (or set AI_HOST, or use --llm fake)")
    live = build(a)
    start_camera(live, None if a.no_camera else a.camera)
    print(f"open http://127.0.0.1:{a.port}")
    make_app(live).run(host="127.0.0.1", port=a.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
