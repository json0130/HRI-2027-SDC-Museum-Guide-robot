"""Process-wide "museum mode is on" switch that makes disk persistence refuse to run.

Museum visits are ephemeral: no face embeddings, transcripts or KG person data may reach disk.
Rather than trusting callers, the persistence entry points (``SessionStore(path)``,
``FaceIdentifier.save/load``, ``FirstImpression.auto_enroll/learn_name``, ``GraphStore.save``)
call ``forbid()`` first. With no active visit the call is a no-op, so existing modes behave
exactly as before. ``modules.museum.profile.VisitProfile`` calls ``enter()`` on creation and
``leave()`` in ``end()``; the switch is ref-counted, so overlapping visits are fine.

This guards the code paths in this repo. It cannot stop arbitrary file writes.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_depth = 0


class EphemeralViolation(RuntimeError):
    """Raised when something tries to persist visitor data during a museum visit."""


def enter() -> None:
    global _depth
    with _lock:
        _depth += 1


def leave() -> None:
    global _depth
    with _lock:
        _depth = max(0, _depth - 1)


def reset() -> None:
    """Test helper: drop all active visits."""
    global _depth
    with _lock:
        _depth = 0


def active() -> bool:
    return _depth > 0


def forbid(what: str) -> None:
    if _depth > 0:
        raise EphemeralViolation(f"{what} is disabled during a museum visit (visits are ephemeral)")
