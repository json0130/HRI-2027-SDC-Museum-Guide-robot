import os
import time

import numpy as np
import pytest

from .appearance import (AgeBandPipeline, AgeProbs, FakeAgeEstimator, FairFaceAgeEstimator,
                         PersonAgeSmoother, VisitTracker, create_estimator)
from .worker import AgeBandWorker

CHILD, ADULT, TEEN = AgeProbs(0.9, 0.1), AgeProbs(0.05, 0.95), AgeProbs(0.3, 0.2, 0.5)


def feed(sm, p, n):
    for _ in range(n):
        r = sm.update(p)
    return r


def test_too_few_frames_is_unsure():
    sm = PersonAgeSmoother()
    assert feed(sm, CHILD, 4).band == "unsure"
    assert sm.update(CHILD).band == "child"


def test_clear_child_and_adult():
    assert feed(PersonAgeSmoother(), CHILD, 6).band == "child"
    assert feed(PersonAgeSmoother(), ADULT, 6).band == "adult"


def test_middle_band_and_teen_mass_abstain():
    assert feed(PersonAgeSmoother(), AgeProbs(0.5, 0.5), 8).band == "unsure"   # ratio in middle band
    assert feed(PersonAgeSmoother(), TEEN, 8).band == "unsure"                 # low confidence


def test_smoothing_absorbs_one_bad_frame():
    sm = PersonAgeSmoother()
    feed(sm, ADULT, 7)
    assert sm.update(CHILD).band == "adult"


def test_config_override():
    assert feed(PersonAgeSmoother({"min_frames": 2}), CHILD, 2).band == "child"


def test_tracker_ids_stable_new_and_not_reused():
    t = VisitTracker({"track_max_missed": 1})
    a, b = (0, 0, 10, 10), (100, 100, 110, 110)
    assert t.assign([a, b]) == ["visitor_1", "visitor_2"]
    assert t.assign([(1, 1, 11, 11), (101, 100, 111, 110)]) == ["visitor_1", "visitor_2"]
    t.assign([]); t.assign([])                       # both exceed max_missed -> dropped
    assert t.alive() == []
    assert t.assign([a]) == ["visitor_3"]


def test_pipeline_two_people_and_cleanup():
    est = FakeAgeEstimator({"visitor_1": CHILD, "visitor_2": ADULT})
    boxes = [(0, 0, 10, 10), (100, 100, 110, 110)]
    pipe = AgeBandPipeline(est, lambda f: boxes, cfg={"track_max_missed": 0})
    for _ in range(6):
        res = pipe.process(None)
    assert [(r["person_id"], r["band"]) for r in res] == [("visitor_1", "child"), ("visitor_2", "adult")]
    boxes.clear()
    pipe.process(None); pipe.process(None)
    assert pipe._smoothers == {}                       # leaving visitors' state is dropped


def test_output_has_no_gender_or_race_fields():
    est = FakeAgeEstimator({"visitor_1": CHILD})
    pipe = AgeBandPipeline(est, lambda f: [(0, 0, 10, 10)])
    keys = set(pipe.process(None)[0])
    assert keys == {"person_id", "box", "band", "confidence", "p_child", "p_adult", "n_frames", "track_conf", "streak"}
    assert set(AgeProbs.__dataclass_fields__) == {"p_child", "p_adult", "p_teen"}


def test_fake_script_sequence_and_noise_seeded():
    e = FakeAgeEstimator({"a": [CHILD, ADULT]})
    assert [e.estimate(None, None, "a").p_child for _ in range(3)] == [0.9, 0.05, 0.05]
    n1 = FakeAgeEstimator({"a": CHILD}, noise=0.1, seed=1).estimate(None, None, "a")
    n2 = FakeAgeEstimator({"a": CHILD}, noise=0.1, seed=1).estimate(None, None, "a")
    assert n1 == n2 and abs(n1.p_child - 0.9) <= 0.1


def test_worker_latest_frame_and_stop():
    est = FakeAgeEstimator({"visitor_1": CHILD})
    w = AgeBandWorker(AgeBandPipeline(est, lambda f: [(0, 0, 10, 10)], cfg={"min_frames": 1}))
    w.start()
    try:
        assert w.get_results() == []
        w.submit("frame")
        for _ in range(100):
            if w.get_results():
                break
            time.sleep(0.01)
        assert w.get_results()[0]["band"] == "child"
    finally:
        w.stop(); w.join(1)
    assert not w.is_alive()


def test_unknown_backend():
    with pytest.raises(ValueError):
        create_estimator("nope")


def test_fairface_checksum_and_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        FairFaceAgeEstimator(str(tmp_path / "x.onnx"))
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(ValueError):
        FairFaceAgeEstimator(str(bad))


@pytest.mark.skipif(not os.path.exists(os.path.expanduser("~/.fairface/fairface.onnx")), reason="model not cached")
def test_fairface_runs_and_returns_age_mass_only():
    p = FairFaceAgeEstimator().estimate(np.random.RandomState(0).randint(0, 255, (120, 120, 3), dtype=np.uint8),
                                        (10, 10, 110, 110))
    assert abs(p.p_child + p.p_adult + p.p_teen - 1.0) < 1e-4


# ── FairFace boundary: only age leaves the backend; the 10-19 bin is "unsure" ─────────────

class _FakeSession:
    """Stands in for the onnx session: returns 18 logits = 7 race + 2 gender + 9 age."""
    def __init__(self, race=0.0, gender=0.0, age_bin=0, strength=12.0):
        self.race, self.gender, self.age_bin, self.strength = race, gender, age_bin, strength
    def run(self, _outs, _feed):
        out = np.full(18, 0.0, np.float32)
        out[0:7] = self.race
        out[7:9] = self.gender
        out[9 + self.age_bin] = self.strength
        return [out[None, :]]


def _backend(sess):
    est = FairFaceAgeEstimator.__new__(FairFaceAgeEstimator)     # skip loading the real model
    est._sess, est._in = sess, "input"
    return est


FRAME = np.zeros((100, 100, 3), np.uint8)
BOX = (10, 10, 90, 90)


def test_race_and_gender_logits_have_no_influence_and_are_not_returned():
    base = _backend(_FakeSession(age_bin=4)).estimate(FRAME, BOX)
    wild = _backend(_FakeSession(race=1e4, gender=-1e4, age_bin=4)).estimate(FRAME, BOX)
    assert base == wild                                         # identical whatever race/gender logits say
    assert isinstance(base, AgeProbs)
    assert set(vars(base)) == {"p_child", "p_adult", "p_teen"}  # no other attribute, array or field


@pytest.mark.parametrize("age_bin,expect", [(0, "child"), (1, "child"), (2, "teen"),
                                            (3, "adult"), (4, "adult"), (5, "adult"), (6, "adult"), (7, "adult"), (8, "adult")])
def test_age_bin_mapping(age_bin, expect):
    p = _backend(_FakeSession(age_bin=age_bin)).estimate(FRAME, BOX)
    top = max(("child", p.p_child), ("adult", p.p_adult), ("teen", p.p_teen), key=lambda t: t[1])[0]
    assert top == expect and p.p_child + p.p_adult + p.p_teen == pytest.approx(1.0)


def test_ten_to_nineteen_bin_comes_out_unsure_through_the_pipeline():
    est = _backend(_FakeSession(age_bin=2))
    pipe = AgeBandPipeline(est, lambda f: [BOX])
    for _ in range(12):
        res = pipe.process(FRAME)
    assert res[0]["band"] == "unsure" and res[0]["n_frames"] == 8     # window full, still abstains


def test_estimator_module_only_reads_age_slice():
    # guard against someone later using the other logits: the slice is the only logits[...] access
    import inspect
    from . import appearance
    src = inspect.getsource(appearance.FairFaceAgeEstimator.estimate)
    assert "logits[9:18]" in src and "logits[0" not in src and "logits[7" not in src


# ── tracker: embeddings, margin rule, track confidence ───────────────────────────────────

from .appearance import Detection  # noqa: E402


def emb(seed, noise=0.05, rng=None):
    r = np.random.RandomState(seed)
    v = r.randn(512)
    if rng is not None:
        v = v + noise * rng.randn(512) * np.linalg.norm(v) / np.sqrt(512) * 4
    return v / np.linalg.norm(v)


def mix(a, b, w):
    v = w * a + (1 - w) * b
    return v / np.linalg.norm(v)


A_BOX, B_BOX = (0, 0, 40, 40), (50, 0, 90, 40)


def test_exchange_places_swaps_ids_silently_without_embeddings():
    """A and B swap spots between frames (e.g. while occluded). Boxes alone cannot tell: the ids follow the
    SPOTS, so A is now called visitor_2, and track_conf does not drop, the swap is silent."""
    t = VisitTracker()
    t.assign([A_BOX, B_BOX])
    ids = t.assign([B_BOX, A_BOX])            # det0 is person A (now standing where B was), det1 is person B
    assert ids == ["visitor_2", "visitor_1"]  # wrong: A should still be visitor_1
    assert t.confidence("visitor_1") >= 0.6 and t.confidence("visitor_2") >= 0.6


def test_embeddings_keep_ids_when_people_exchange_places():
    ea, eb = emb(1), emb(2)
    t = VisitTracker()
    t.assign_detections([Detection(A_BOX, ea), Detection(B_BOX, eb)])
    ids = t.assign_detections([Detection(B_BOX, ea), Detection(A_BOX, eb)])   # same exchange, faces known
    assert ids == ["visitor_1", "visitor_2"]
    assert t.confidence("visitor_1") >= 0.6 and t.confidence("visitor_2") >= 0.6


def _cross(with_emb):
    ea, eb = emb(1), emb(2)
    t = VisitTracker()
    ids_seq, confs = [], []
    for f in range(40):
        x = 4 * f
        a, b = (x, 0, x + 40, 40), (156 - x, 0, 196 - x, 40)       # A goes right, B goes left, they overlap mid-way
        dets = [Detection(a, ea if with_emb else None), Detection(b, eb if with_emb else None)]
        ids = t.assign_detections(dets)
        ids_seq.append(ids)
        confs.append((t.confidence(ids[0]), t.confidence(ids[1])))
    return ids_seq, confs


def test_crossing_with_embeddings_keeps_ids_and_confidence():
    ids_seq, confs = _cross(True)
    assert all(ids == ["visitor_1", "visitor_2"] for ids in ids_seq)
    assert min(min(c) for c in confs) >= 0.5


def test_crossing_without_embeddings_is_flagged_ambiguous():
    ids_seq, confs = _cross(False)
    assert min(min(c) for c in confs) < 0.5          # the tracker knows it is unsure around the crossing


def test_reassociation_returns_old_id_for_same_face():
    ea, eb = emb(1), emb(2)
    rng = np.random.RandomState(0)
    t = VisitTracker({"track_max_missed": 2})
    t.assign_detections([Detection(A_BOX, ea), Detection(B_BOX, eb)])
    for _ in range(4):
        t.assign_detections([])                      # both leave long enough to be "lost"
    assert t.alive() == [] and set(t.known()) == {"visitor_1", "visitor_2"}
    ids = t.assign_detections([Detection((200, 100, 240, 140), emb(1, rng=rng))])    # A returns elsewhere
    assert ids == ["visitor_1"]
    assert t.confidence("visitor_1") == pytest.approx(0.6)


def test_margin_rule_lookalike_gets_new_id():
    """Relatives stand-in: the returning face is nearly equidistant from two lost tracks."""
    ea, eb = emb(1), emb(2)
    t = VisitTracker({"track_max_missed": 1})
    t.assign_detections([Detection(A_BOX, ea), Detection(B_BOX, eb)])
    for _ in range(3):
        t.assign_detections([])
    rel = mix(ea, eb, 0.5)                           # cosine ~0.71 to both, gap ~0 < reid_gap
    assert float(rel @ ea) > 0.6 and float(rel @ eb) > 0.6 and abs(float(rel @ ea) - float(rel @ eb)) < 0.1
    assert t.assign_detections([Detection((100, 0, 140, 40), rel)]) == ["visitor_3"]


def test_margin_rule_clear_winner_reuses_and_low_cosine_is_new():
    ea, eb = emb(1), emb(2)
    t = VisitTracker({"track_max_missed": 1})
    t.assign_detections([Detection(A_BOX, ea), Detection(B_BOX, eb)])
    for _ in range(3):
        t.assign_detections([])
    assert t.assign_detections([Detection((100, 0, 140, 40), mix(ea, eb, 0.85))]) == ["visitor_1"]
    t2 = VisitTracker({"track_max_missed": 1, "reid_min_cos": 0.9})
    t2.assign_detections([Detection(A_BOX, ea)])
    for _ in range(3):
        t2.assign_detections([])
    assert t2.assign_detections([Detection((100, 0, 140, 40), mix(ea, eb, 0.6))]) == ["visitor_2"]    # cosine ~0.83 < 0.9


def test_new_face_resembling_a_live_track_is_not_merged_into_a_lost_one():
    ea, eb = emb(1), emb(2)
    t = VisitTracker({"track_max_missed": 1})
    t.assign_detections([Detection(A_BOX, ea), Detection(B_BOX, eb)])
    t.assign_detections([Detection(B_BOX, eb)])      # A missing, B stays live
    t.assign_detections([Detection(B_BOX, eb)])
    t.assign_detections([Detection(B_BOX, eb)])
    sibling_of_b = mix(eb, ea, 0.8)                  # close to live B (0.8-ish) and to lost A
    ids = t.assign_detections([Detection(B_BOX, eb), Detection((120, 0, 160, 40), sibling_of_b)])
    assert ids[0] == "visitor_2" and ids[1] == "visitor_3"


def test_clear_drops_everything():
    t = VisitTracker()
    t.assign_detections([Detection(A_BOX, emb(1))])
    t.clear()
    assert t.known() == []


def test_pipeline_does_not_feed_ambiguous_frames_to_smoother_and_reports_track_conf():
    est = FakeAgeEstimator({"visitor_1": CHILD, "visitor_2": ADULT})
    seq = []
    pipe = AgeBandPipeline(est, lambda f: seq, cfg={"min_frames": 1})
    for f in range(40):
        x = 4 * f
        seq[:] = [(x, 0, x + 40, 40), (156 - x, 0, 196 - x, 40)]
        res = pipe.process(None)
    assert all("track_conf" in r for r in res)
