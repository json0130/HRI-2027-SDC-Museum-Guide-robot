"""Coarse visual first impression for museum mode: age band ONLY (child | adult | unsure).

Hard rules (see modules/museum/README.md):
  * Only an age band is estimated. Gender, race/ethnicity and personality are never
    estimated, stored or returned. A backend that happens to output them (FairFace does) must
    discard them inside the backend; nothing outside this file ever sees them. (An earlier
    ChatBox branch rejected appearance-based culture/ethnicity inference as unreliable.)
  * The result is a weak prior for how the robot opens. It is never spoken and never written
    as a profile fact.
  * Nothing here touches disk except the optional read of the model file; face crops stay in
    memory and visitor tracks are dropped when the visitor leaves.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .config import load_config

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2

DEFAULTS = {
    "window": 8, "min_frames": 5, "min_conf": 0.6,
    "abstain_lo": 0.35, "abstain_hi": 0.65,
    "track_iou": 0.3, "track_max_missed": 30,
}


@dataclass(frozen=True)
class AgeProbs:
    """Mass over three outcomes (sums to ~1). ``p_teen`` is the ambiguous 10-19 bin; it counts
    toward neither child nor adult, so a teenager lowers confidence and comes out 'unsure'."""
    p_child: float
    p_adult: float
    p_teen: float = 0.0


@dataclass(frozen=True)
class AgeBand:
    band: str            # "child" | "adult" | "unsure"
    confidence: float    # 0..1, only meaningful when band != "unsure"
    p_child: float
    p_adult: float
    n_frames: int


class AgeBandEstimator(Protocol):
    def estimate(self, frame, box: Box, key: Optional[str] = None) -> AgeProbs: ...


# ── backends ──────────────────────────────────────────────────────────────────

class FakeAgeEstimator:
    """Scripted backend for tests/simulation. ``script[key]`` is an AgeProbs (repeated) or a
    list consumed in order (last element repeats). Optional seeded jitter on p_child."""

    def __init__(self, script: Dict[str, object], noise: float = 0.0, seed: int = 0):
        self._script = {k: (list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in script.items()}
        self._pos: Dict[str, int] = {}
        self._noise = noise
        self._rng = random.Random(seed)

    def estimate(self, frame, box: Box, key: Optional[str] = None) -> AgeProbs:
        seq = self._script[key]
        i = self._pos.get(key, 0)
        self._pos[key] = i + 1
        p = seq[min(i, len(seq) - 1)]
        if self._noise:
            d = self._rng.uniform(-self._noise, self._noise)
            pc = min(max(p.p_child + d, 0.0), 1.0 - p.p_teen)
            return AgeProbs(pc, max(1.0 - p.p_teen - pc, 0.0), p.p_teen)
        return p


_FAIRFACE_PATH = os.path.join(os.path.expanduser("~"), ".fairface", "fairface.onnx")
# SHA-256 of the file this was developed against (third-party ONNX conversion of FairFace,
# HF garavv/fairface-onnx). A mismatch refuses to load rather than trusting a swapped file.
_FAIRFACE_SHA256 = "e26179a4125c0c51fa73676f4b38b2013e9abe7d20ccc4ca19192dc9e2825971"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# FairFace age bins: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
_CHILD_BINS, _TEEN_BINS, _ADULT_BINS = (0, 1), (2,), (3, 4, 5, 6, 7, 8)


class FairFaceAgeEstimator:
    """FairFace ResNet-34 via onnxruntime (CPU). Output is 18 logits: 7 race + 2 gender + 9 age.
    Only the 9 age logits [9:18] are read; the race/gender logits are discarded here, never
    softmaxed, never returned. Age bins are coarse and accuracy varies across groups: treat as
    a weak prior. The model file is not downloaded automatically."""

    def __init__(self, model_path: str = _FAIRFACE_PATH, verify_sha: bool = True):
        import onnxruntime as ort  # lazy: costs nothing when museum mode is off
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FairFace model not found at {model_path}")
        if verify_sha:
            h = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != _FAIRFACE_SHA256:
                raise ValueError("FairFace model checksum mismatch; refusing to load")
        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._in = self._sess.get_inputs()[0].name

    @staticmethod
    def _crop(frame_bgr: np.ndarray, box: Box, pad: float = 0.25) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        return frame_bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]

    def estimate(self, frame, box: Box, key: Optional[str] = None) -> AgeProbs:
        import cv2
        face = self._crop(frame, box)
        rgb = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA)
        x = ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        logits = self._sess.run(None, {self._in: x})[0][0]
        age = logits[9:18]                      # race [0:7] and gender [7:9] deliberately ignored
        e = np.exp(age - age.max())
        p = e / e.sum()
        return AgeProbs(float(p[list(_CHILD_BINS)].sum()), float(p[list(_ADULT_BINS)].sum()),
                        float(p[list(_TEEN_BINS)].sum()))


def create_estimator(backend: str, **kw) -> AgeBandEstimator:
    if backend == "fake":
        return FakeAgeEstimator(**kw)
    if backend == "fairface":
        return FairFaceAgeEstimator(**kw)
    raise ValueError(f"unknown age backend {backend!r}")


# ── smoothing ─────────────────────────────────────────────────────────────────

class PersonAgeSmoother:
    """Rolling mean of AgeProbs for ONE tracked person -> AgeBand. Abstains ('unsure') with too
    few frames, low confidence, or a child/adult ratio inside the middle band."""

    def __init__(self, cfg: Optional[dict] = None):
        c = {**DEFAULTS, **(cfg or {})}
        self._c = c
        self._buf: deque = deque(maxlen=int(c["window"]))

    def update(self, p: AgeProbs) -> AgeBand:
        self._buf.append(p)
        n = len(self._buf)
        pc = sum(q.p_child for q in self._buf) / n
        pa = sum(q.p_adult for q in self._buf) / n
        conf = max(pc, pa)
        ratio = pc / (pc + pa) if pc + pa > 0 else 0.5
        if n < self._c["min_frames"] or conf < self._c["min_conf"] or self._c["abstain_lo"] < ratio < self._c["abstain_hi"]:
            return AgeBand("unsure", conf, pc, pa, n)
        return AgeBand("child" if ratio >= self._c["abstain_hi"] else "adult", conf, pc, pa, n)


def _iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class VisitTracker:
    """In-memory IoU tracker assigning visit-scoped ids ``visitor_N``. No embeddings, nothing
    persisted; ids are never reused within a visit. ponytail: IoU only, ids can swap when
    people cross or leave and re-enter; add an in-memory appearance match if that bites."""

    def __init__(self, cfg: Optional[dict] = None):
        c = {**DEFAULTS, **(cfg or {})}
        self._iou, self._max_missed = c["track_iou"], int(c["track_max_missed"])
        self._tracks: Dict[str, List] = {}   # id -> [box, missed]
        self._n = 0

    def assign(self, boxes: Sequence[Box]) -> List[str]:
        pairs = sorted(((_iou(t[0], b), tid, i) for tid, t in self._tracks.items() for i, b in enumerate(boxes)),
                       reverse=True)
        ids: List[Optional[str]] = [None] * len(boxes)
        used = set()
        for s, tid, i in pairs:
            if s < self._iou:
                break
            if tid in used or ids[i] is not None:
                continue
            ids[i], _ = tid, used.add(tid)
            self._tracks[tid] = [boxes[i], 0]
        for i, b in enumerate(boxes):
            if ids[i] is None:
                self._n += 1
                ids[i] = f"visitor_{self._n}"
                self._tracks[ids[i]] = [b, 0]
                used.add(ids[i])
        for tid in list(self._tracks):
            if tid not in used:
                self._tracks[tid][1] += 1
                if self._tracks[tid][1] > self._max_missed:
                    del self._tracks[tid]
        return ids  # type: ignore[return-value]

    def alive(self) -> List[str]:
        return list(self._tracks)


class AgeBandPipeline:
    """Detect boxes -> track -> estimate -> smooth. Synchronous; the worker thread just calls
    ``process``. ``detect_fn(frame) -> [Box]`` is injected (e.g. wrapping FaceIdentifier's
    detector) so no face is ever enrolled."""

    def __init__(self, estimator: AgeBandEstimator, detect_fn: Callable, cfg: Optional[dict] = None):
        self.cfg = {**DEFAULTS, **(cfg if cfg is not None else load_config("museum_appearance", DEFAULTS))}
        self._est, self._detect = estimator, detect_fn
        self._tracker = VisitTracker(self.cfg)
        self._smoothers: Dict[str, PersonAgeSmoother] = {}

    def process(self, frame) -> List[dict]:
        boxes = list(self._detect(frame))
        ids = self._tracker.assign(boxes)
        out = []
        for pid, box in zip(ids, boxes):
            sm = self._smoothers.setdefault(pid, PersonAgeSmoother(self.cfg))
            r = sm.update(self._est.estimate(frame, box, key=pid))
            out.append({"person_id": pid, "box": box, "band": r.band, "confidence": r.confidence,
                        "p_child": r.p_child, "p_adult": r.p_adult, "n_frames": r.n_frames})
        alive = set(self._tracker.alive())
        for pid in [p for p in self._smoothers if p not in alive]:
            del self._smoothers[pid]
        return out
