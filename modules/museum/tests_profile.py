import json
import os

import pytest

from modules.graph_relationship.schema import PersonNode, TopicNode
from modules.graph_relationship.store import InMemoryGraphStore

from .profile import VisitProfile
from .profile_extraction import extract_and_apply_profile


def make():
    p = VisitProfile("v1")
    p.add_person("visitor_1")
    return p


# ── schema back-compat ───────────────────────────────────────────────────────

def test_old_kg_state_loads_and_new_types_round_trip(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"nodes": [PersonNode(id="jay").model_dump(mode="json"),
                                         TopicNode(id="t:x", label="x").model_dump(mode="json")], "edges": []}))
    s = InMemoryGraphStore()
    assert s.load(str(old), quiet=True) and s.get_node("jay") is not None

    p = make()
    p.record_style("visitor_1", "pace", -1, "choice_answer", turn_ref=2)
    p.record_topic("visitor_1", "dinosaurs", "animals", +1, "llm_statement")
    out = tmp_path / "new.json"
    # store.save() now refuses during a visit (see tests_ephemeral), so serialise by hand, same format
    out.write_text(json.dumps({"nodes": [n.model_dump(mode="json") for n in p.store._nodes.values()],
                               "edges": [e.model_dump(mode="json") for e in p.store._edges.values()]}))
    s2 = InMemoryGraphStore()
    assert s2.load(str(out), quiet=True)
    e = s2.get_edge("visitor_1", "style:pace", "visit_pref")
    assert e.value == -1.0 and e.visit_id == "v1" and e.evidence[0].turn_ref == 2


# ── evidence maths ───────────────────────────────────────────────────────────

def test_weights_confidence_and_shrinkage():
    p = make()
    p.record_style("visitor_1", "talk_level", +1, "behavior_cue")
    s = p.style("visitor_1", "talk_level")
    assert s["confidence"] == pytest.approx(0.25) and s["effective_value"] == pytest.approx(0.25)
    p.record_style("visitor_1", "talk_level", +1, "choice_answer")
    s = p.style("visitor_1", "talk_level")
    assert s["confidence"] == pytest.approx(1 - 0.75 * 0.2) and s["evidence_count"] == 2


def test_contradiction_pulls_value_toward_neutral():
    p = make()
    p.record_style("visitor_1", "depth", +1, "llm_statement")
    p.record_style("visitor_1", "depth", -1, "llm_statement")
    assert p.style("visitor_1", "depth")["value"] == pytest.approx(0.0)


def test_scale_confidence_and_untouched_dim_neutral():
    p = make()
    p.record_style("visitor_1", "pace", +1, "choice_answer")
    p.scale_confidence("visitor_1", "pace", 0.5)
    assert p.style("visitor_1", "pace")["confidence"] == pytest.approx(0.4)
    assert p.style("visitor_1", "depth") == {"value": 0.0, "confidence": 0.0, "effective_value": 0.0, "evidence_count": 0}


def test_likes_and_dislikes_by_sign():
    p = make()
    p.record_topic("visitor_1", "Dinosaurs", "animals", +1, "choice_answer")
    p.record_topic("visitor_1", "war", "other", -1, "llm_statement")
    t = p.topics("visitor_1")
    assert [r["label"] for r in t["likes"]] == ["Dinosaurs"] and [r["label"] for r in t["dislikes"]] == ["war"]


def test_bad_inputs_rejected():
    p = make()
    with pytest.raises(ValueError): p.record_style("visitor_1", "mood", 1, "choice_answer")
    with pytest.raises(ValueError): p.record_style("visitor_1", "pace", 1, "visual_prior")
    with pytest.raises(KeyError): p.record_style("visitor_9", "pace", 1, "choice_answer")


# ── visual prior is not a profile fact ───────────────────────────────────────

def test_prior_decays_and_never_enters_kg():
    p = make()
    p.set_prior("visitor_1", "child", 0.9)
    w0 = p.prior("visitor_1")["weight"]
    assert w0 == pytest.approx(0.1) and p.topics("visitor_1") == {"likes": [], "dislikes": []}
    assert all(p.style("visitor_1", d)["evidence_count"] == 0 for d in ("talk_level", "depth", "pace"))
    p.record_style("visitor_1", "pace", 1, "behavior_cue")
    assert p.prior("visitor_1")["weight"] == pytest.approx(0.05)
    assert not any("child" in json.dumps(e.model_dump(mode="json")) for e in p.store._edges.values())


# ── extraction guards (fake LLM) ─────────────────────────────────────────────

GOOD = json.dumps({"topics": [{"label": "dinosaurs", "category": "animals", "sign": "like", "confidence": 0.9,
                               "quote": "love dinosaurs"}],
                   "style": [{"dim": "depth", "direction": "high", "confidence": 0.8, "quote": "full story"}]})
TEXT = "I love dinosaurs, give me the full story"


def test_extraction_applies_good_items():
    p = make()
    r = extract_and_apply_profile(p, "visitor_1", TEXT, lambda s, u: GOOD, turn_ref=3)
    assert r["applied"] and (r["topics"], r["style"]) == (1, 1)
    assert p.style("visitor_1", "depth")["confidence"] == pytest.approx(0.5)
    assert p.topics("visitor_1")["likes"][0]["label"] == "dinosaurs"


@pytest.mark.parametrize("raw", ["not json", "", "[1,2]", '{"topics":"x"}', '{"topics":[],"style":5}'])
def test_parse_or_shape_failure_applies_nothing(raw):
    p = make()
    r = extract_and_apply_profile(p, "visitor_1", TEXT, lambda s, u: raw)
    assert not r["applied"] and p.store._edges == {}


def test_item_guards_drop_bad_items_keep_good():
    bad = json.dumps({"topics": [
        {"label": "x", "category": "weird", "sign": "like", "confidence": 0.9, "quote": "dinosaurs"},       # vocab
        {"label": "y", "category": "animals", "sign": "like", "confidence": 0.3, "quote": "dinosaurs"},     # low conf
        {"label": "z", "category": "animals", "sign": "like", "confidence": 0.9, "quote": "robots"},        # hallucinated
        {"label": "dinosaurs", "category": "animals", "sign": "like", "confidence": 0.9, "quote": "love dinosaurs"}],
        "style": [{"dim": "mood", "direction": "high", "confidence": 0.9, "quote": "full story"}]})
    p = make()
    r = extract_and_apply_profile(p, "visitor_1", TEXT, lambda s, u: bad)
    assert (r["topics"], r["style"]) == (1, 0) and len(r["dropped"]) == 4


def test_llm_exception_and_unknown_person_are_safe():
    def boom(s, u): raise TimeoutError
    p = make()
    assert not extract_and_apply_profile(p, "visitor_1", TEXT, boom)["applied"]
    assert not extract_and_apply_profile(p, "nobody", TEXT, lambda s, u: GOOD)["applied"]
    assert p.store._edges == {}


def test_prompt_has_no_mood_field():
    from .profile_extraction import build_system_prompt
    assert "mood" in build_system_prompt().lower()            # only in the "do NOT infer" clause
    assert '"mood"' not in build_system_prompt()


# ── ephemeral: wipe at end, nothing on disk ──────────────────────────────────

def test_end_wipes_everything_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))
    p = make()
    p.add_person("visitor_2")
    extract_and_apply_profile(p, "visitor_1", TEXT, lambda s, u: GOOD)
    p.record_style("visitor_2", "pace", 1, "choice_answer")
    assert p.group_size == 2
    r = p.end()
    assert r["persons_deleted"] == 2 and r["topics_deleted"] == 1
    assert p.store._nodes == {} and p.store._edges == {} and p.persons == []
    assert set(os.listdir(tmp_path)) == before
    with pytest.raises(RuntimeError):
        p.record_style("visitor_1", "pace", 1, "choice_answer")
