"""Coarse group context for the opening: who is probably here together.

Output per visitor: ``GroupContext(group_id, group_size, has_child_band_member, status)``. That is ALL. Relationship labels
(couple, family, parent/child, friends) are never inferred: that is appearance-based guessing of social facts, the same
class of risk as the culture inference an earlier branch rejected.

Same rules as the age prior: weak, opening-only, never spoken, never stored as a profile fact (it lives in memory next to
the prior, not in the KG), hidden from get_visit_snapshot by default, and it falls back to ``unknown`` whenever membership
is unstable.

Membership = proximity AND co-movement, held over time:
  close    edge-to-edge gap <= gap_widths x the LARGER of the two face widths (so a child's small face does not make a
           family look farther apart than an adult pair);
  comove   the two tracks first appeared within arrival_s of each other, and their headings agree (both standing still,
           or both moving with cos(angle) >= heading_cos). One still + one moving is NOT co-moving (a passer-by).
A pair is "together" when close AND co-moving for >= together_s within window_s. Groups are connected components of
"together" among stable tracks. Any change in membership (join, leave, split) makes the context unknown until it has been
stable for stable_s; a merge of two groups, or a component larger than max_group_size, reports unknown (merge: for
merge_hold_s). A lone visitor is also unknown until stable_s after they arrived: companions may still be walking up.

Limits (placeholders, untuned): image distance is not real distance (a camera angle can fake togetherness); people
queueing side by side look together; occlusions longer than window_s dissolve a group; the age bands that feed
has_child_band_member are themselves weak.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Tuple

from .config import load_config

DEFAULTS = {
    "gap_widths": 1.5, "together_s": 3.0, "window_s": 10.0, "arrival_s": 4.0, "heading_window_s": 2.0,
    "still_widths_per_s": 0.25, "heading_cos": 0.7, "stable_s": 1.5, "merge_hold_s": 10.0, "max_group_size": 6,
    "stable_frames": 5,
}


@dataclass(frozen=True)
class GroupContext:
    group_id: Optional[str]
    group_size: Optional[int]                    # None when unknown
    has_child_band_member: Optional[bool]        # True / False / None (unknown)
    status: str                                  # "known" | "unknown"
    reason: str = ""                             # settling | merge | cap | no_track | ""

    def to_dict(self) -> dict:
        return asdict(self)


UNKNOWN = GroupContext(None, None, None, "unknown", "no_track")


def _geom(box) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, float(x2 - x1), float(y2 - y1)


def _gap(a, b) -> float:
    """Edge-to-edge distance between two boxes (0 if they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


class GroupTracker:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = {**DEFAULTS, **(cfg if cfg is not None else load_config("museum_group", DEFAULTS))}
        self.clear()

    def clear(self) -> None:
        self._hist: Dict[str, Deque] = {}
        self._arrival: Dict[str, float] = {}
        self._box: Dict[str, tuple] = {}
        self._band: Dict[str, str] = {}
        self._streak: Dict[str, int] = {}
        self._pairs: Dict[Tuple[str, str], Deque] = {}
        self._groups: Dict[str, dict] = {}
        self._n = 0
        self._ctx: Dict[str, GroupContext] = {}

    # -- co-movement ----------------------------------------------------------

    def _velocity(self, pid: str) -> Tuple[float, float, float]:
        """(vx, vy, speed) in FACE WIDTHS per second over heading_window_s."""
        h = self._hist[pid]
        if len(h) < 2:
            return 0.0, 0.0, 0.0
        t0, x0, y0, w0 = h[0]
        t1, x1, y1, w1 = h[-1]
        dt, w = max(t1 - t0, 1e-6), max((w0 + w1) / 2.0, 1.0)
        vx, vy = (x1 - x0) / dt / w, (y1 - y0) / dt / w
        return vx, vy, math.hypot(vx, vy)

    def _comove(self, a: str, b: str) -> bool:
        c = self.cfg
        if abs(self._arrival[a] - self._arrival[b]) > c["arrival_s"]:
            return False
        va, vb = self._velocity(a), self._velocity(b)
        still_a, still_b = va[2] < c["still_widths_per_s"], vb[2] < c["still_widths_per_s"]
        if still_a and still_b:
            return True
        if still_a != still_b:                      # one standing, one moving: a passer-by, not a companion
            return False
        return (va[0] * vb[0] + va[1] * vb[1]) / (va[2] * vb[2]) >= c["heading_cos"]

    def _close(self, a: str, b: str) -> bool:
        wmax = max(_geom(self._box[a])[2], _geom(self._box[b])[2])      # the LARGER face width
        return _gap(self._box[a], self._box[b]) <= self.cfg["gap_widths"] * wmax

    # -- update ---------------------------------------------------------------

    def update(self, t: float, results: List[dict]) -> Dict[str, GroupContext]:
        """`results` = AgeBandPipeline.process() output (needs person_id, box, band, streak). Returns the context of
        every stable visitor currently in a component."""
        c = self.cfg
        present = []
        for r in results:
            pid = r["person_id"]
            cx, cy, w, _ = _geom(r["box"])
            h = self._hist.setdefault(pid, deque())
            h.append((t, cx, cy, w))
            while h and t - h[0][0] > c["heading_window_s"]:
                h.popleft()
            self._arrival.setdefault(pid, t)
            self._box[pid], self._band[pid], self._streak[pid] = r["box"], r["band"], r["streak"]
            present.append(pid)
        members = [p for p in present if self._streak[p] >= c["stable_frames"]]
        # pair evidence: sample only when both are present now; old samples age out of window_s
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                key = (a, b) if a < b else (b, a)
                self._pairs.setdefault(key, deque()).append((t, self._close(a, b) and self._comove(a, b)))
        together = []
        for key, dq in list(self._pairs.items()):
            while dq and t - dq[0][0] > c["window_s"]:
                dq.popleft()
            if not dq:
                del self._pairs[key]; continue
            credit, prev = 0.0, None
            for ts, ok in dq:
                if ok and prev is not None:
                    credit += min(ts - prev, 0.5)
                prev = ts
            if credit >= c["together_s"]:
                together.append(key)
        # candidates: stable members now, plus anyone still bound to them by unexpired pair evidence
        cand = set(members) | {p for k in together for p in k}
        parent = {p: p for p in cand}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for a, b in together:
            parent[find(a)] = find(b)
        comps: Dict[str, set] = {}
        for p in cand:
            comps.setdefault(find(p), set()).add(p)
        components = list(comps.values())
        # identity of groups across frames
        overlaps = sorted(((len(comp & g["members"]), gi, ci) for ci, comp in enumerate(components) for gi, g in self._groups.items()
                           if comp & g["members"]), reverse=True)
        assigned_comp, taken, new_groups = {}, set(), {}
        for _, gi, ci in overlaps:
            if ci in assigned_comp or gi in taken:
                continue
            assigned_comp[ci] = gi; taken.add(gi)
        for ci, comp in enumerate(components):
            # a merge = two ESTABLISHED (once-known) groups of 2+ became one. A lone visitor joining someone, or fresh
            # singletons pairing up, is a group forming (membership change -> settling), not a merge.
            n_prior = sum(1 for g in self._groups.values()
                          if comp & g["members"] and g.get("established") and len(g["members"]) >= 2)
            if ci in assigned_comp:
                gid, g = assigned_comp[ci], dict(self._groups[assigned_comp[ci]])
                if g["members"] != comp:
                    g["changed_at"] = t
                if n_prior >= 2:                               # two groups became one
                    g["hold_until"] = t + c["merge_hold_s"]
            else:
                self._n += 1
                gid, g = f"group_{self._n}", {"changed_at": t, "hold_until": 0.0, "established": False}
            g["members"] = set(comp)
            new_groups[gid] = g
        self._groups = new_groups
        # contexts
        self._ctx = {}
        for gid, g in self._groups.items():
            comp, n = g["members"], len(g["members"])
            latest_arrival = max(self._arrival[p] for p in comp)
            if n > c["max_group_size"]:
                ctx = GroupContext(gid, None, None, "unknown", "cap")
            elif t < g["hold_until"]:
                ctx = GroupContext(gid, None, None, "unknown", "merge")
            elif t - max(g["changed_at"], latest_arrival) < c["stable_s"]:
                ctx = GroupContext(gid, None, None, "unknown", "settling")
            elif n == 1 and any(q != next(iter(comp)) and self._close(next(iter(comp)), q) and self._comove(next(iter(comp)), q)
                                for q in present):
                # a lone visitor with someone close and co-moving who has not yet accumulated together_s: a companion
                # may be arriving, so a solo context would be premature
                ctx = GroupContext(gid, None, None, "unknown", "settling")
            else:
                bands = [self._band.get(p, "unsure") for p in comp]
                child = True if "child" in bands else (False if all(b == "adult" for b in bands) else None)
                ctx = GroupContext(gid, n, child, "known", "")
                g["established"] = True
            for p in comp:
                self._ctx[p] = ctx
        return dict(self._ctx)

    def context(self, pid: str) -> GroupContext:
        return self._ctx.get(pid, UNKNOWN)

    def members(self, group_id: str) -> List[str]:
        g = self._groups.get(group_id)
        return sorted(g["members"]) if g else []
