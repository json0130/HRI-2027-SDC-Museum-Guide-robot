"""Steps 1-2: one id per visit, the visit-scoped KG node, and its lifecycle (all synthetic: they show the logic works,
not that real faces separate)."""
import json
import os

import numpy as np
import pytest

from modules import ephemeral
from modules.graph_relationship.schema import PersonNode
from modules.graph_relationship.store import InMemoryGraphStore

from .appearance import AgeBandPipeline, AgeProbs, Detection, FakeAgeEstimator
from .decision_log import DecisionLog, snapshot
from .profile import VisitProfile

A = (0, 0, 40, 40)
ADULT = AgeProbs(0.05, 0.95)


def emb(seed):
    v = np.random.RandomState(seed).randn(512)
    return v / np.linalg.norm(v)


def pipeline(frames, cfg=None):
    """frames: list of lists of Detection (one list per frame)."""
    it = iter(frames)
    est = FakeAgeEstimator({f"visitor_{i}": ADULT for i in range(1, 6)})
    return AgeBandPipeline(est, lambda f: next(it), cfg=cfg or {})


def test_old_person_node_without_visit_id_still_loads(tmp_path):
    old = tmp_path / "old.json"
    n = PersonNode(id="jay").model_dump(mode="json"); n.pop("visit_id")
    old.write_text(json.dumps({"nodes": [n], "edges": []}))
    s = InMemoryGraphStore()
    assert s.load(str(old), quiet=True) and s.get_node("jay").visit_id is None


def test_node_is_created_only_after_n_consecutive_frames_and_carries_visit_id():
    vp = VisitProfile("v-1")
    frames = [[Detection(A, emb(1))]] * 4
    pipe = pipeline(frames)
    created = []
    for _ in range(4):
        created += vp.sync_tracks(pipe.process(None), stable_frames=5)
    assert created == [] and vp.persons == []                    # 4 consecutive frames < 5: no node yet
    pipe2 = pipeline([[Detection(A, emb(1))]] * 5)
    for _ in range(5):
        created += vp.sync_tracks(pipe2.process(None), stable_frames=5)
    assert created == ["visitor_1"] and vp.persons == ["visitor_1"]
    assert vp.store.get_node("visitor_1").visit_id == "v-1"
    vp.end()


def test_a_broken_streak_does_not_count_and_a_passer_by_never_becomes_a_node():
    vp = VisitProfile("v-1")
    seq = [[Detection(A, emb(1))]] * 3 + [[]] + [[Detection(A, emb(1))]] * 3        # 3 + gap + 3: never 5 in a row
    pipe = pipeline(seq)
    for _ in range(len(seq)):
        vp.sync_tracks(pipe.process(None), stable_frames=5)
    assert vp.persons == []
    ghost = [[Detection(A, emb(1)), Detection((300, 0, 340, 40), emb(2))]] + [[Detection(A, emb(1))]] * 6    # 2nd face: 1 frame
    pipe = pipeline(ghost)
    for _ in range(len(ghost)):
        vp.sync_tracks(pipe.process(None), stable_frames=5)
    assert vp.persons == ["visitor_1"]                                              # the one-frame face never got a node
    vp.end()


def test_end_deletes_nodes_edges_and_wipes_the_embedding_gallery():
    vp = VisitProfile("v-1")
    pipe = pipeline([[Detection(A, emb(1))]] * 6)
    vp.add_cleanup(pipe.clear)
    for _ in range(6):
        vp.sync_tracks(pipe.process(None), stable_frames=5)
    vp.record_style("visitor_1", "pace", 1, "choice_answer")
    assert pipe.tracker.known() == ["visitor_1"] and vp.store._edges
    vp.end()
    assert vp.store._nodes == {} and vp.store._edges == {}
    assert pipe.tracker.known() == [] and pipe._smoothers == {}                     # embeddings gone, nothing left to re-identify


def test_writes_dropped_counted_and_logged_while_track_conf_is_low(tmp_path):
    vp = VisitProfile("v-1")
    vp.add_person("visitor_1")
    vp.set_track_conf("visitor_1", 0.1)
    vp.record_style("visitor_1", "pace", 1, "choice_answer"); vp.record_topic("visitor_1", "x", "other", 1, "llm_statement")
    log = DecisionLog(str(tmp_path), "profile", clock=lambda: 0.0)
    log.log("visitor_1", prior=None, profile=snapshot(vp, "visitor_1"), action={"kind": "none"}, gate="none", outcome="none",
            reversal=False, track_conf=0.1, suspended_writes=vp.suspended_writes)
    log.close()
    assert json.loads(open(log.path).read())["suspended_writes"] == 2 and vp.store._edges == {}
    vp.end()


def test_end_to_end_tracking_visit_in_an_empty_dir_creates_only_the_decision_log(tmp_path, monkeypatch):
    work = tmp_path / "empty"; work.mkdir(); monkeypatch.chdir(work)
    vp = VisitProfile("v-1")
    seq = [[Detection(A, emb(1)), Detection((300, 0, 340, 40), emb(2))]] * 8
    pipe = pipeline(seq)
    vp.add_cleanup(pipe.clear)
    log = DecisionLog(str(work / "logs"), "profile")
    for _ in range(8):
        res = pipe.process(None)
        vp.sync_tracks(res, stable_frames=5)
    for pid in vp.persons:
        log.log(pid, prior={"band": "adult", "confidence": 0.9}, profile=snapshot(vp, pid), action={"kind": "none"}, gate="none",
                outcome="none", reversal=False, track_conf=vp.is_suspended(pid) and 0.0 or 1.0, suspended_writes=vp.suspended_writes)
    log.close()
    vp.end()
    files = [os.path.join(r, f) for r, _, fs in os.walk(work) for f in fs]
    assert len(files) == 1 and files[0].endswith(".jsonl")                          # no faces.npz, sessions.db, kg_state.json, embeddings
    assert not ephemeral.active()
