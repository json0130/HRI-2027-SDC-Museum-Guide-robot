"""Anonymised per-visit decision log (JSONL): the ONLY thing museum mode keeps after a visit.

One file ``<log_dir>/<visit_id>.jsonl``; ``visit_id`` is random (not derived from time, place or
people) and ``t`` is seconds since the visit started. Each line is whitelisted field by field:
no images, no embeddings, no names, no visitor speech, and topics only as closed category counts
(never labels). Person ids are the visit-scoped ``visitor_N``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Callable, Dict, Optional

from .profile import DIMS, VisitProfile

MODES = ("fixed", "prior-only", "profile")


def snapshot(profile: VisitProfile, person_id: str) -> dict:
    """Closed-vocab, label-free view of a person's profile."""
    topics = profile.topics(person_id)
    cats: Dict[str, int] = {}
    for k in ("likes", "dislikes"):
        for r in topics[k]:
            cats[f"{k[:-1]}:{r['category']}"] = cats.get(f"{k[:-1]}:{r['category']}", 0) + 1
    return {"dims": {d: {k: v for k, v in profile.style(person_id, d).items() if k != "value"} for d in DIMS},
            "topics_by_category": cats}


class DecisionLog:
    def __init__(self, log_dir: str, mode: str, clock: Optional[Callable[[], float]] = None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.visit_id = uuid.uuid4().hex[:12]
        self.mode = mode
        self._clock = clock or time.monotonic
        self._t0 = self._clock()
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{self.visit_id}.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, person: str, *, prior: Optional[dict], profile: dict, action: dict, gate: str,
            outcome: str, reversal: bool, track_conf: float, suspended_writes: int) -> None:
        pr = None if prior is None else {"band": prior["band"], "confidence": round(prior["confidence"], 3)}
        row = {"t": round(self._clock() - self._t0, 3), "visit_id": self.visit_id, "mode": self.mode, "person": person,
               "prior": pr, "profile": profile,
               "action": {"kind": action.get("kind"), "dim": action.get("dim"), "direction": action.get("direction"),
                          "level": action.get("level")},
               "gate": gate, "outcome": outcome, "reversal": bool(reversal),
               "track_conf": round(float(track_conf), 3), "suspended_writes": int(suspended_writes)}
        self._fh.write(json.dumps(row, sort_keys=True) + "\n")
        self._fh.flush()

    def log_event(self, kind: str, *, persons: list, group: Optional[dict] = None, distance_proxy: Optional[float] = None,
                  opening_id: Optional[str] = None, prior: Optional[dict] = None, track_conf: Optional[float] = None,
                  suspended_writes: int = 0, reason: str = "") -> None:
        """Engagement-level events: kind is engage | disengaged | disengaged_returned. Same whitelist rules: visit-scoped ids,
        closed-vocabulary fields only, no images, embeddings, names or text."""
        g = None if group is None else {k: group.get(k) for k in ("group_id", "group_size", "has_child_band_member", "status")}
        pr = None if prior is None else {"band": prior["band"], "confidence": round(prior["confidence"], 3)}
        row = {"t": round(self._clock() - self._t0, 3), "visit_id": self.visit_id, "mode": self.mode, "event": kind,
               "persons": list(persons), "group": g, "distance_proxy": None if distance_proxy is None else round(float(distance_proxy), 3),
               "opening_id": opening_id, "prior": pr, "track_conf": None if track_conf is None else round(float(track_conf), 3),
               "suspended_writes": int(suspended_writes), "reason": reason}
        self._fh.write(json.dumps(row, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
