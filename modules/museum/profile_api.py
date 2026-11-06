"""Read-only interface for the group-level policy (teammate). JSON-serialisable dicts only.

No group aggregation lives here. Start a visit with ``start_visit``; museum code writes through the
returned ``VisitProfile``; consumers call ``get_person_profile`` / ``get_visit_snapshot``.
"""

from __future__ import annotations

from typing import Dict, Optional

from .profile import DIMS, VisitProfile

_VISITS: Dict[str, VisitProfile] = {}


def start_visit(visit_id: str, cfg: Optional[dict] = None) -> VisitProfile:
    if visit_id in _VISITS:
        raise ValueError(f"visit {visit_id!r} already started")
    _VISITS[visit_id] = VisitProfile(visit_id, cfg)
    return _VISITS[visit_id]


def end_visit(visit_id: str) -> dict:
    """Wipe the visit (persons, edges, topics) and forget it. Returns counts only."""
    return _VISITS.pop(visit_id).end()


def get_person_profile(visit_id: str, person_id: str, include_prior: bool = False, include_group: bool = False) -> dict:
    vp = _VISITS[visit_id]
    if person_id not in vp.persons:
        raise KeyError(person_id)
    t = vp.topics(person_id)
    out = {
        "visit_id": visit_id,
        "person_id": person_id,
        "dimensions": {d: {**vp.style(person_id, d), "low": lo, "high": hi} for d, (lo, hi) in DIMS.items()},
        "likes": t["likes"],
        "dislikes": t["dislikes"],
    }
    if include_prior:
        # Opening-move hint from the camera, NOT a preference and not evidence. Present only when
        # asked for, marked "use": "opening_only", and None once any real evidence exists.
        out["prior"] = vp.prior(person_id)
    if include_group:
        # Coarse group context (group_id / size / has_child_band_member), opening-only like the prior; hidden by default,
        # None once any real evidence exists, and never a relationship label.
        out["group"] = vp.group_context(person_id)
    return out


def get_visit_snapshot(visit_id: str, include_prior: bool = False, include_group: bool = False) -> dict:
    vp = _VISITS[visit_id]
    return {"visit_id": visit_id, "group_size": vp.group_size,
            "persons": {p: get_person_profile(visit_id, p, include_prior, include_group) for p in vp.persons}}
