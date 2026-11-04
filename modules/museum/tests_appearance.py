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
    assert keys == {"person_id", "box", "band", "confidence", "p_child", "p_adult", "n_frames"}
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
