"""Visit-scoped per-person profile in the KG: likes, dislikes and style preferences.

Everything lives in a private InMemoryGraphStore that is never ``save()``d. Person ids are
``visitor_N``. ``end()`` deletes the visit's person nodes, profile edges and the topic nodes the
visit created; nothing is written to disk at any point.

Evidence model (config/museum_profile.yaml): each item (a topic like/dislike or a style
dimension) is a ProfileEdge holding a signed value, a confidence and an evidence list.
  confidence = noisy-OR of evidence weights; value = weight-weighted mean of signed evidence.
  ``effective_value = value * confidence`` pulls low-confidence items toward neutral; that is the
  only number downstream code should act on.

The visual age-band prior is deliberately NOT evidence: it is held outside the KG, is dropped
entirely once any real evidence arrives, and is never exposed as a dimension.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from modules.graph_relationship.schema import (
    Evidence, PersonNode, ProfileEdge, Provenance, StyleDimNode, TOPIC_CATEGORIES,
)
from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.topics import resolve_topic

from modules import ephemeral

from .config import load_config

# dim -> (value -1 meaning, value +1 meaning)
DIMS: Dict[str, tuple] = {"talk_level": ("quiet", "chatty"), "depth": ("brief", "detailed"), "pace": ("slow", "fast")}
SOURCES = ("choice_answer", "llm_statement", "behavior_cue")

DEFAULTS = {
    "weights": {"choice_answer": 0.8, "llm_statement": 0.5, "behavior_cue": 0.25},
    "extraction": {"max_items_per_turn": 3},
    "min_track_conf": 0.5,
}
_EDGE = "visit_pref"


def _style_id(dim: str) -> str:
    return f"style:{dim}"


class VisitProfile:
    def __init__(self, visit_id: str, cfg: Optional[dict] = None):
        self.visit_id = visit_id
        self.cfg = cfg if cfg is not None else load_config("museum_profile", DEFAULTS)
        self.store = InMemoryGraphStore()
        self._persons: List[str] = []
        self._prior: Dict[str, dict] = {}
        self._n_real: Dict[str, int] = {}
        self._topics: Set[str] = set()
        self._track_conf: Dict[str, float] = {}
        self.suspended_writes = 0
        self.suspended_by_person: Dict[str, int] = {}
        self._cleanups: List = []
        self._group: Dict[str, dict] = {}
        self.ended = False
        ephemeral.enter()     # from here until end(), disk persistence of visitor data refuses to run

    # -- persons / prior ------------------------------------------------------

    def add_person(self, person_id: str) -> None:
        if person_id not in self._persons:
            self.store.upsert_node(PersonNode(id=person_id, visit_id=self.visit_id))
            self._persons.append(person_id)
            self._n_real[person_id] = 0

    @property
    def persons(self) -> List[str]:
        return list(self._persons)

    @property
    def group_size(self) -> int:
        return len(self._persons)

    def set_group_context(self, person_id: str, ctx) -> None:
        """Weak, opening-only group context (modules.museum.group.GroupContext). Like the age prior it is held in memory,
        never written to the KG, dropped once the person has real evidence, and wiped by end()."""
        self._group[person_id] = ctx.to_dict() if hasattr(ctx, "to_dict") else dict(ctx)

    def group_context(self, person_id: str) -> Optional[dict]:
        g = self._group.get(person_id)
        if g is None or self._n_real.get(person_id, 0) > 0:
            return None
        return {**g, "use": "opening_only"}

    def set_track_conf(self, person_id: str, conf: float) -> None:
        """Identity confidence of this person's face track (from the tracker). Below
        cfg['min_track_conf'] new evidence is DROPPED, not buffered (see ``is_suspended``)."""
        self._track_conf[person_id] = float(conf)

    def add_cleanup(self, fn) -> None:
        """Run `fn()` when the visit ends (e.g. ``pipeline.clear`` to wipe the in-memory face-embedding gallery)."""
        self._cleanups.append(fn)

    def sync_tracks(self, results, stable_frames: int = 5) -> List[str]:
        """Feed AgeBandPipeline results in. Creates the visit-scoped person node only once a track has been matched
        in `stable_frames` CONSECUTIVE frames, so passers-by and one-frame detections never become nodes. Also updates
        track confidence. Returns the ids of nodes created by this call."""
        new = []
        for r in results:
            if r["streak"] >= stable_frames and r["person_id"] not in self._persons:
                self.add_person(r["person_id"]); new.append(r["person_id"])
        self.apply_tracks(results)
        return new

    def apply_tracks(self, results) -> None:
        """Feed AgeBandPipeline.process() results in: updates track confidence per person."""
        for r in results:
            if r["person_id"] in self._persons:
                self.set_track_conf(r["person_id"], r["track_conf"])

    def is_suspended(self, person_id: str) -> bool:
        return self._track_conf.get(person_id, 1.0) < self.cfg["min_track_conf"]

    def set_prior(self, person_id: str, band: str, confidence: float) -> None:
        self._prior[person_id] = {"band": band, "confidence": float(confidence)}

    def prior(self, person_id: str) -> Optional[dict]:
        """The camera's age-band hint, or None. Opening-only, and gone as soon as any real
        evidence (an answer, a statement, a behaviour cue) exists for this person."""
        p = self._prior.get(person_id)
        if p is None or self._n_real.get(person_id, 0) > 0:
            return None
        return {**p, "use": "opening_only"}

    # -- writing evidence -----------------------------------------------------

    def _add(self, person_id: str, target_id: str, sign: int, source: str, turn_ref: Optional[int],
             strength: float) -> Optional[ProfileEdge]:
        if self.ended:
            raise RuntimeError("visit has ended")
        if person_id not in self._persons:
            raise KeyError(f"unknown person {person_id!r}")
        if source not in SOURCES:
            raise ValueError(f"unknown evidence source {source!r}")
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        if self.is_suspended(person_id):      # who this is, is in doubt: write nothing (and count it)
            self.suspended_writes += 1
            self.suspended_by_person[person_id] = self.suspended_by_person.get(person_id, 0) + 1
            return None
        w = max(0.0, min(1.0, self.cfg["weights"][source] * strength))
        ev = Evidence(source=source, weight=w, sign=sign, turn_ref=turn_ref)
        old = self.store.get_edge(person_id, target_id, _EDGE)
        if old is None:
            value, conf, evs = float(sign), w, [ev]
        else:
            w_old = sum(e.weight for e in old.evidence)
            value = (old.value * w_old + sign * w) / (w_old + w) if w_old + w > 0 else 0.0
            conf = 1.0 - (1.0 - old.confidence) * (1.0 - w)
            evs = old.evidence + [ev]
        edge = ProfileEdge(source_id=person_id, target_id=target_id, value=max(-1.0, min(1.0, value)),
                           confidence=max(0.0, min(1.0, conf)), visit_id=self.visit_id, evidence=evs,
                           provenance=Provenance(source=f"museum:{source}", confidence=min(1.0, conf)))
        self._n_real[person_id] += 1
        return self.store.upsert_edge(edge)

    def record_style(self, person_id: str, dim: str, sign: int, source: str, turn_ref: Optional[int] = None,
                     strength: float = 1.0) -> Optional[ProfileEdge]:
        """Returns None when the write was suspended (low track confidence). sign +1 = the high end (chatty/detailed/fast), -1 = the low end (quiet/brief/slow)."""
        if dim not in DIMS:
            raise ValueError(f"unknown style dimension {dim!r}")
        sid = _style_id(dim)
        if self.store.get_node(sid) is None:
            self.store.upsert_node(StyleDimNode(id=sid, dim=dim))
        return self._add(person_id, sid, sign, source, turn_ref, strength)

    def record_topic(self, person_id: str, label: str, category: str, sign: int, source: str,
                     turn_ref: Optional[int] = None, strength: float = 1.0) -> Optional[ProfileEdge]:
        """sign +1 = likes, -1 = dislikes. None when the write was suspended."""
        if category not in TOPIC_CATEGORIES:
            category = "other"
        if person_id not in self._persons:
            raise KeyError(f"unknown person {person_id!r}")
        if self.is_suspended(person_id):
            return self._add(person_id, "", sign, source, turn_ref, strength)   # counted + dropped, no topic node
        t = resolve_topic(self.store, label, category)
        self._topics.add(t.id)
        return self._add(person_id, t.id, sign, source, turn_ref, strength)

    def scale_confidence(self, person_id: str, dim: str, factor: float) -> None:
        """Lower confidence after a reversed step (profiling_policy). No-op if no edge yet."""
        e = self.store.get_edge(person_id, _style_id(dim), _EDGE)
        if e is not None:
            self.store.upsert_edge(e.model_copy(update={"confidence": max(0.0, min(1.0, e.confidence * factor))}))

    # -- reading --------------------------------------------------------------

    @staticmethod
    def _view(e: Optional[ProfileEdge]) -> dict:
        if e is None:
            return {"value": 0.0, "confidence": 0.0, "effective_value": 0.0, "evidence_count": 0}
        return {"value": round(e.value, 4), "confidence": round(e.confidence, 4),
                "effective_value": round(e.value * e.confidence, 4), "evidence_count": len(e.evidence)}

    def style(self, person_id: str, dim: str) -> dict:
        if dim not in DIMS:
            raise ValueError(f"unknown style dimension {dim!r}")
        return self._view(self.store.get_edge(person_id, _style_id(dim), _EDGE))

    def topics(self, person_id: str) -> Dict[str, List[dict]]:
        likes, dislikes = [], []
        for e, n in self.store.query_neighbors(person_id, _EDGE):
            if n.node_type != "topic":
                continue
            row = {"label": n.label, "category": getattr(n.category, "value", n.category), **self._view(e)}
            if e.value * e.confidence > 0:
                likes.append(row)
            elif e.value * e.confidence < 0:
                dislikes.append(row)
        key = lambda r: -abs(r["effective_value"])
        return {"likes": sorted(likes, key=key), "dislikes": sorted(dislikes, key=key)}

    # -- end of visit ---------------------------------------------------------

    def end(self) -> dict:
        """Delete every person node (with its edges) and the topics this visit created."""
        if self.ended:
            return {"visit_id": self.visit_id, "persons_deleted": 0, "topics_deleted": 0}
        n_p, n_t = len(self._persons), len(self._topics)
        for pid in self._persons:
            self.store.delete_node(pid)
        for tid in self._topics:
            self.store.delete_node(tid)
        for dim in DIMS:
            self.store.delete_node(_style_id(dim))
        self._persons, self._prior, self._n_real, self._topics = [], {}, {}, set()
        self._track_conf, self._group = {}, {}
        for fn in self._cleanups:               # e.g. wipe the in-memory face-embedding gallery
            fn()
        self._cleanups = []
        self.ended = True
        ephemeral.leave()
        return {"visit_id": self.visit_id, "persons_deleted": n_p, "topics_deleted": n_t}
