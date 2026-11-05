import os

import pytest

from modules import ephemeral
from modules.ephemeral import EphemeralViolation
from modules.first_impression import FirstImpression
from modules.graph_relationship.store import InMemoryGraphStore
from modules.session_store import SessionStore

from .profile import VisitProfile


class FakeFaces:
    def __init__(self): self.saved = []
    def known_people(self): return []
    def enroll(self, n, f): return True
    def rename(self, a, b): return True
    def save(self, p): self.saved.append(p)


def fi(faces=None):
    return FirstImpression(faces or FakeFaces(), "faces.npz", InMemoryGraphStore(), "robot", SessionStore(":memory:"))


def test_existing_modes_unchanged_when_no_visit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not ephemeral.active()
    SessionStore("s.db")                                   # disk store works as before
    f = FakeFaces()
    assert fi(f).auto_enroll(object()) == "guest_1" and f.saved == ["faces.npz"]
    InMemoryGraphStore().save("kg.json")
    assert {"s.db", "kg.json"} <= set(os.listdir(tmp_path))


def test_museum_visit_refuses_every_disk_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vp = VisitProfile("v1")
    assert ephemeral.active()
    with pytest.raises(EphemeralViolation): SessionStore("sessions.db")
    SessionStore(":memory:")                               # in-memory transcript store is allowed
    f = FakeFaces()
    m = fi(f)
    with pytest.raises(EphemeralViolation): m.auto_enroll(object())
    with pytest.raises(EphemeralViolation): m.learn_name("guest_1", "jay", "Jay")
    with pytest.raises(EphemeralViolation): m.meet(None, "I'm Jay", object())
    with pytest.raises(EphemeralViolation): InMemoryGraphStore().save("kg_state.json")
    assert f.saved == [] and os.listdir(tmp_path) == []   # nothing written, faces.npz never created
    vp.end()
    assert not ephemeral.active()
    SessionStore("sessions.db")                           # back to normal afterwards


def test_real_face_identifier_save_and_load_refuse(tmp_path, monkeypatch):
    from modules.face_webcam import face_id            # hard import: a missing dependency must fail, not skip
    monkeypatch.chdir(tmp_path)
    fid = face_id.FaceIdentifier.__new__(face_id.FaceIdentifier)    # no models loaded
    vp = VisitProfile("v1")
    with pytest.raises(EphemeralViolation): fid.save("faces.npz")
    with pytest.raises(EphemeralViolation): fid.load("faces.npz")
    assert os.listdir(tmp_path) == []
    vp.end()


def test_overlapping_visits_and_double_end():
    a, b = VisitProfile("a"), VisitProfile("b")
    a.end(); a.end()                       # second end is a no-op, must not unbalance the counter
    assert ephemeral.active()
    b.end()
    assert not ephemeral.active()
