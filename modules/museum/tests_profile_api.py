import json

import pytest

from . import profile_api as api


@pytest.fixture(autouse=True)
def clean():
    yield
    for v in list(api._VISITS):
        api.end_visit(v)


def test_snapshot_contract_and_json():
    vp = api.start_visit("v1")
    vp.add_person("visitor_1"); vp.add_person("visitor_2")
    vp.record_style("visitor_1", "depth", +1, "choice_answer")
    vp.record_topic("visitor_1", "dinosaurs", "animals", +1, "llm_statement")
    vp.record_topic("visitor_2", "war", "other", -1, "llm_statement")
    snap = json.loads(json.dumps(api.get_visit_snapshot("v1")))
    assert snap["group_size"] == 2 and set(snap["persons"]) == {"visitor_1", "visitor_2"}
    p1 = snap["persons"]["visitor_1"]
    assert set(p1["dimensions"]) == {"talk_level", "depth", "pace"}
    d = p1["dimensions"]["depth"]
    assert (d["value"], d["confidence"], d["evidence_count"], d["low"], d["high"]) == (1.0, 0.8, 1, "brief", "detailed")
    assert p1["likes"][0]["label"] == "dinosaurs" and snap["persons"]["visitor_2"]["dislikes"][0]["label"] == "war"
    assert api.get_person_profile("v1", "visitor_1") == p1


def test_prior_hidden_by_default_opening_only_when_included_gone_after_evidence():
    vp = api.start_visit("v1")
    vp.add_person("visitor_1"); vp.add_person("visitor_2")
    vp.set_prior("visitor_1", "child", 0.9); vp.set_prior("visitor_2", "adult", 0.8)
    vp.record_topic("visitor_2", "jazz", "music", +1, "llm_statement")
    default = api.get_visit_snapshot("v1")
    assert all("prior" not in p for p in default["persons"].values()) and "child" not in json.dumps(default)
    assert "prior" not in api.get_person_profile("v1", "visitor_1")
    inc = api.get_visit_snapshot("v1", include_prior=True)["persons"]
    assert inc["visitor_1"]["prior"] == {"band": "child", "confidence": 0.9, "use": "opening_only"}
    assert inc["visitor_2"]["prior"] is None                          # real evidence exists -> dropped
    assert all(v["evidence_count"] == 0 for v in inc["visitor_1"]["dimensions"].values())
    assert "child" not in json.dumps(inc["visitor_1"]["dimensions"])


def test_unknown_ids_and_end_visit_wipes():
    vp = api.start_visit("v1")
    vp.add_person("visitor_1")
    with pytest.raises(KeyError): api.get_person_profile("v1", "nobody")
    with pytest.raises(KeyError): api.get_visit_snapshot("nope")
    with pytest.raises(ValueError): api.start_visit("v1")
    assert api.end_visit("v1")["persons_deleted"] == 1
    with pytest.raises(KeyError): api.get_visit_snapshot("v1")
