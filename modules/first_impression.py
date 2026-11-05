"""
First impression: meet a stranger.

Auto-enrol an unrecognised face as a provisional ``guest_N``, then re-key that guest to
their real name across every store once they introduce themselves.

Lifted from ChatBox-companion-robot (``modules/face_webcam/webcam_loop.py``: ``_extract_name``,
``_slug_name``, ``_next_guest_id``, ``_auto_enroll``, ``_learn_name`` and the chat-Enter
handler). The logic is unchanged; the methods now live on a small class that takes the stores
as arguments instead of reaching into the 3,000-line ``WebcamKGLoop``.
"""

from __future__ import annotations

import re
import threading
from typing import Callable, Iterable, Optional

import numpy as np

from modules import ephemeral

# ─────────────────────────────────────────────────────────────────────────────
# Name extraction (guard: feelings / filler / culture words are not names)
# ─────────────────────────────────────────────────────────────────────────────

_NAME_RE = re.compile(
    r"\b(?:my name is|my name's|i am|i'm|im|call me|this is|name is|name's|names)"
    r"\s+([A-Za-z][A-Za-z'\-]{1,19})",
    re.IGNORECASE,
)
# Words that follow the trigger phrases but are NOT names — avoids "I'm fine" /
# "I am not sure" being read as the name "fine" / "not".
_NAME_STOPWORDS = frozenset({
    "not", "sorry", "fine", "good", "great", "okay", "ok", "here", "just",
    "really", "so", "very", "doing", "the", "a", "an", "sure", "happy", "sad",
    "tired", "hungry", "back", "done", "trying", "going", "feeling", "glad",
    "curious", "bored", "excited", "confused", "afraid", "scared", "korean", "maori",
})


def extract_name(text: Optional[str]) -> Optional[str]:
    """Pull a first name out of a self-introduction, or None. Rejects obvious
    non-names (feelings/filler) via a small stoplist."""
    m = _NAME_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1).strip("-'")
    if len(raw) < 2 or raw.lower() in _NAME_STOPWORDS:
        return None
    return raw


def slug_name(name: str) -> str:
    """Lowercase single-token id from a captured name (faces keyed 'jay', 'alice')."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Enrol + learn name
# ─────────────────────────────────────────────────────────────────────────────

class FirstImpression:
    """
    face_id        FaceIdentifier-like: known_people(), enroll(name, frame), rename(a, b), save(path)
    faces_path     where the face DB is persisted
    store          GraphStore (graph_relationship) — person/interaction/conversation nodes
    robot_id       the robot's node id in `store`
    session_store  SessionStore — transcript rows are re-keyed on rename
    store_lock     held around graph writes (a chat worker may be reading concurrently)
    rekey_maps     per-person dicts keyed by person id, re-keyed on rename (last mood, history…)
    on_change      called after a successful rename (e.g. mark the KG dirty so it gets saved)
    """

    def __init__(self, face_id, faces_path: str, store, robot_id: str, session_store,
                 store_lock: Optional[threading.RLock] = None,
                 rekey_maps: Iterable[dict] = (),
                 on_change: Optional[Callable[[], None]] = None):
        self.face_id = face_id
        self.faces_path = faces_path
        self.store = store
        self.robot_id = robot_id
        self._session_store = session_store
        self._store_lock = store_lock or threading.RLock()
        self._rekey_maps = list(rekey_maps)
        self._on_change = on_change
        self._guest_seq = 0

    def next_guest_id(self) -> str:
        """Next free provisional id ('guest_1', 'guest_2', …) not already in the face
        DB — so re-runs don't collide with earlier guests."""
        known = set(self.face_id.known_people())
        self._guest_seq += 1
        while f"guest_{self._guest_seq}" in known:
            self._guest_seq += 1
        return f"guest_{self._guest_seq}"

    def auto_enroll(self, frame: np.ndarray) -> Optional[str]:
        """Enrol the face in `frame` under a fresh provisional id and persist the face
        DB. Returns the new id, or None if no face was captured."""
        ephemeral.forbid("first-impression auto-enrol (it saves faces.npz)")
        if frame is None:
            print("[FirstImpression] auto-enrol skipped — no face seen recently")
            return None
        gid = self.next_guest_id()
        if not self.face_id.enroll(gid, frame):
            self._guest_seq -= 1          # release the id we reserved
            print("[FirstImpression] auto-enrol: couldn't capture a clean face — "
                  "try again facing the camera")
            return None
        self.face_id.save(self.faces_path)
        print(f"[FirstImpression] new face → enrolled as '{gid}' (guest)")
        return gid

    def learn_name(self, old_id: str, new_id: str, display: str) -> bool:
        """Re-key a provisional guest to their real name across the face DB, the graph,
        the transcript store and the in-memory maps."""
        ephemeral.forbid("first-impression name learning (it saves faces.npz)")
        if not self.face_id.rename(old_id, new_id):
            return False
        self.face_id.save(self.faces_path)
        try:
            from modules.graph_relationship.rename import rename_person
            with self._store_lock:
                rename_person(self.store, old_id, new_id, self.robot_id,
                              display_name=display)
        except Exception as exc:  # noqa: BLE001 — never crash the loop on rename
            print(f"[FirstImpression] graph rename failed: {exc}")
        self._session_store.rename_person(old_id, new_id)
        for d in self._rekey_maps:
            if old_id in d:
                d[new_id] = d.pop(old_id)
        if self._on_change:
            self._on_change()
        print(f"[FirstImpression] learned name: '{old_id}' → '{new_id}'")
        return True

    def meet(self, person_id: Optional[str], msg: str,
             last_face_frame: Optional[np.ndarray]) -> Optional[str]:
        """The chat-turn hook: call before replying. Returns the person id this turn
        belongs to (possibly a new guest_N, or the real name once they introduce themselves)."""
        # (1) Unrecognised person → auto-enrol as guest_N so THIS turn is attributed to them.
        if person_id is None:
            if last_face_frame is None:
                print("[FirstImpression] no face seen yet — look at the camera so I can meet you")
                return None
            person_id = self.auto_enroll(last_face_frame)
        # (2) Did they introduce themselves? Re-key the provisional guest to their real name.
        if person_id and person_id.startswith("guest_"):
            nm = extract_name(msg)
            slug = slug_name(nm) if nm else ""
            if slug and slug != person_id and self.learn_name(person_id, slug, nm):
                person_id = slug
        return person_id
