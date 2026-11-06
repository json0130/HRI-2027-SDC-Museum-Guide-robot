"""Engagement trigger: start the first interaction once a visitor is close enough. Robot-agnostic: it only emits an
``EngageEvent``; a Pepper (or other) adapter consumes it later and the caller then runs ``opening_for_event``.

Distance proxy = face-box height / frame height. This is NOT a distance. Known limitation: a child's face is smaller than an
adult's, so at the same real distance the box ratio is lower and the trigger fires later (or not at all) for children. This is
deliberately NOT corrected with the age band; Pepper's depth sensing should replace the proxy.

Rules:
  * A person counts toward dwell only when their KG node exists, ``track_conf`` >= min_track_conf, and the ratio is >=
    enter_ratio. The count resets below exit_ratio and HOLDS between exit and enter (hysteresis); it resets when the person is
    absent or not eligible. Passers-by never reach ``dwell_frames``.
  * The first member to pass the dwell test starts a settle wait: if their group context becomes known the event fires at
    once for the whole group (one event, not one per member: the opening is for the group, and a smaller-faced child member
    does not delay or split it); if group_settle_s passes with the context still unknown the event fires with the people who
    have passed dwell so far and the context unknown (the caller then uses the neutral opening).
  * Each person gets at most max_attempts_per_person events (default 1), at most one event per cooldown_s.
  * After an event, a person who is absent for leave_s before ``exchange_started`` is DISENGAGED for the rest of the visit:
    never re-greeted. If they come back it is logged, nothing is triggered. (If the tracker issues them a new id on return
    the engine cannot know they are the same person: a documented limit.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import load_config
from .group import GroupContext, UNKNOWN
from .initiation import OpeningParams, choose_opening

DEFAULTS = {
    "enter_ratio": 0.20, "exit_ratio": 0.15, "dwell_frames": 20, "min_track_conf": 0.6, "cooldown_s": 30.0,
    "max_attempts_per_person": 1, "group_settle_s": 4.0, "leave_s": 3.0,
}


@dataclass(frozen=True)
class EngageEvent:
    person_ids: Tuple[str, ...]
    group_id: Optional[str]
    distance_proxy: float          # box-height ratio of the first member to pass dwell
    t: float
    group: GroupContext
    reason: str                    # "group_known" | "settle_timeout"


class EngagementEngine:
    def __init__(self, group_ctx: Callable[[str], GroupContext], group_members: Callable[[str], List[str]],
                 cfg: Optional[dict] = None):
        self.cfg = {**DEFAULTS, **(cfg if cfg is not None else load_config("museum_engagement", DEFAULTS))}
        self._ctx, self._members = group_ctx, group_members
        self._st: Dict[str, dict] = {}
        self._pending: Optional[dict] = None
        self._last_event_t = -1e9
        self.events_log: List[dict] = []

    def _p(self, pid: str) -> dict:
        return self._st.setdefault(pid, {"state": "idle", "count": 0, "attempts": 0, "absent_since": None, "ratio": 0.0, "returned_logged": False})

    def state(self, pid: str) -> str:
        return self._st[pid]["state"] if pid in self._st else "idle"

    def drain_log(self) -> List[dict]:
        out, self.events_log = self.events_log, []
        return out

    def exchange_started(self, person_ids: Sequence[str], t: float) -> None:
        for pid in person_ids:
            self._p(pid)["state"] = "exchange"

    def update(self, t: float, frame_h: float, observations: List[dict]) -> List[EngageEvent]:
        """observations: one dict per visitor currently seen: {person_id, box, track_conf, eligible} where `eligible` means
        the visit-scoped node exists. Returns the events that fire on this frame (zero or one)."""
        c = self.cfg
        seen = {o["person_id"]: o for o in observations}
        for pid, p in self._st.items():                                   # absent people
            if pid in seen:
                continue
            p["count"] = 0
            if p["state"] == "engaged":
                p["absent_since"] = p["absent_since"] if p["absent_since"] is not None else t
                if t - p["absent_since"] >= c["leave_s"]:
                    p["state"] = "disengaged"
                    self.events_log.append({"kind": "disengaged", "person": pid, "t": t})
        for pid, o in seen.items():
            p = self._p(pid)
            b = o["box"]
            p["ratio"] = ratio = (b[3] - b[1]) / float(frame_h)
            p["absent_since"] = None
            if p["state"] == "disengaged":
                if not p["returned_logged"]:
                    p["returned_logged"] = True
                    self.events_log.append({"kind": "disengaged_returned", "person": pid, "t": t})
                continue
            if p["state"] not in ("idle", "counting") or not o.get("eligible", True) or o["track_conf"] < c["min_track_conf"]:
                if p["state"] in ("idle", "counting"):
                    p["count"], p["state"] = 0, "idle"
                continue
            if ratio >= c["enter_ratio"]:
                p["count"] += 1; p["state"] = "counting"
            elif ratio < c["exit_ratio"]:
                p["count"], p["state"] = 0, "idle"
            # between exit and enter: hold
        # pending group settle
        if self._pending is not None:
            first = self._pending["first"]
            fp = self._st.get(first)
            if fp is None or fp["state"] != "counting" or fp["count"] == 0 or first not in seen:
                self._pending = None                                       # they left the zone before the group settled
            else:
                ctx = self._ctx(first)
                if ctx.status == "known":
                    return self._fire(t, ctx, [m for m in self._members(ctx.group_id) if m in seen] or [first], "group_known")
                if t >= self._pending["deadline"]:
                    ready = [pid for pid, q in self._st.items() if q["state"] == "counting" and q["count"] >= c["dwell_frames"] and pid in seen]
                    return self._fire(t, UNKNOWN if ctx.status != "known" else ctx, ready or [first], "settle_timeout")
                return []
        if t - self._last_event_t < c["cooldown_s"]:
            return []
        for pid, p in self._st.items():
            if p["state"] == "counting" and p["count"] >= c["dwell_frames"] and p["attempts"] < c["max_attempts_per_person"] and pid in seen:
                self._pending = {"first": pid, "deadline": t + c["group_settle_s"], "ratio": p["ratio"]}
                return self.update_pending_now(t, seen)
        return []

    def update_pending_now(self, t: float, seen: Dict[str, dict]) -> List[EngageEvent]:
        first = self._pending["first"]
        ctx = self._ctx(first)
        if ctx.status == "known":
            return self._fire(t, ctx, [m for m in self._members(ctx.group_id) if m in seen] or [first], "group_known")
        return []

    def _fire(self, t: float, ctx: GroupContext, person_ids: List[str], reason: str) -> List[EngageEvent]:
        first_ratio = self._pending["ratio"] if self._pending else 0.0
        self._pending = None
        ids = [pid for pid in person_ids
               if self._p(pid)["state"] in ("idle", "counting") and self._p(pid)["attempts"] < self.cfg["max_attempts_per_person"]]
        if not ids:
            return []
        for pid in ids:
            p = self._p(pid)
            p["state"], p["attempts"], p["absent_since"] = "engaged", p["attempts"] + 1, None
        self._last_event_t = t
        ev = EngageEvent(tuple(ids), ctx.group_id if ctx.status == "known" else None, round(first_ratio, 3), t, ctx, reason)
        self.events_log.append({"kind": "engage", "persons": list(ids), "group_id": ev.group_id, "t": t, "reason": reason})
        return [ev]


def opening_for_event(event: EngageEvent, profile, cfg: Optional[dict] = None) -> OpeningParams:
    """choose_opening at the moment of the event, with the prior and group context AS THEY ARE THEN. Unknown group context ->
    the plain neutral opening. For a solo visitor the first member's age-band prior picks the row; for a known group of 2+ only
    has_child_band_member matters (see choose_opening)."""
    pr = profile.prior(event.person_ids[0]) or {"band": "unsure", "confidence": 0.0}
    return choose_opening(pr["band"], pr["confidence"], len(event.person_ids), group=event.group, cfg=cfg)
