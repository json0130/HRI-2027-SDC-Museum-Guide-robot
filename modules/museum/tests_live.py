import json
import os

import numpy as np
import pytest

from modules import ephemeral
from modules.ephemeral import EphemeralViolation
from modules.session_store import SessionStore

from . import live as L
from .appearance import AgeProbs, Detection, FakeAgeEstimator


def make(band="adult", llm=None, log_dir=None):
    probs = {"child": AgeProbs(0.9, 0.08, 0.02), "adult": AgeProbs(0.06, 0.92, 0.02), "unsure": AgeProbs(0.4, 0.4, 0.2)}[band]
    emb = np.random.RandomState(7).randn(512)
    det = lambda f: [Detection((100, 60, 240, 220), emb / np.linalg.norm(emb))]
    fake_llm = llm or (lambda s, u: json.dumps({"topics": [], "style": []}))
    replies = []
    def reply(system, text): replies.append(system); return "ok"
    live = L.Live(None, fake_llm, FakeAgeEstimator({"visitor_1": probs}), det, log_dir=log_dir, reply_fn=reply)
    live.replies = replies
    return live


def meet(live):
    for _ in range(6):
        live.process_frame(None)


def test_first_met_person_gets_opening_from_prior_without_mentioning_age():
    live = make("child"); meet(live)
    m = live.messages[0]
    assert m["who"] == "robot" and "child_solo" in m["meta"]
    assert not any(w in m["text"].lower() for w in ("young", "kid", "child", "age", "old"))
    assert live.met["visitor_1"].vocabulary == "simple"
    live.end()


def test_unsure_prior_gets_neutral_opening():
    live = make("unsure"); meet(live)
    assert "neutral_solo" in live.messages[0]["meta"]
    live.end()


def test_conversation_builds_profile_and_graph_live_and_adapts_stepwise():
    def llm(system, text):
        t = text.lower()
        if "dinosaurs" in t:
            return json.dumps({"topics": [{"label": "dinosaurs", "category": "animals", "sign": "like", "confidence": 0.9, "quote": "love dinosaurs"}], "style": []})
        return json.dumps({"topics": [], "style": []})
    live = make("adult", llm); meet(live)
    live.chat("The full story please")                      # answers the opening depth question (deterministic, choice_answer)
    st = live.state()
    d = st["persons"]["visitor_1"]["dimensions"]["depth"]
    assert d["value"] > 0 and d["evidence_count"] == 1 and st["persons"]["visitor_1"]["prior"] is None   # real evidence: hint dropped
    live.chat("I love dinosaurs")
    st = live.state()
    assert st["persons"]["visitor_1"]["likes"][0]["label"] == "dinosaurs"
    kinds = {(n["type"]) for n in st["graph"]["nodes"]}
    assert {"person", "style_dim", "topic"} <= kinds and any(e["target"] == "style:depth" for e in st["graph"]["edges"])
    assert live.policy.level("visitor_1", "depth") == 1           # one explicit answer = confident enough for ONE step
    prev = 1
    for _ in range(6):
        live.chat("go on")
        now = live.policy.level("visitor_1", "depth")
        assert abs(now - prev) <= 1 and abs(now) <= 2             # never more than one level per turn, capped
        prev = now
    assert all("adult" not in s.lower() and "child" not in s.lower() for s in live.replies)   # the prior's band never reaches the prompt
    live.end()


def test_negative_reaction_to_adaptation_reverts_and_is_logged():
    live = make("adult"); meet(live)
    live.chat("The full story please")                        # confident -> ONE adaptation step, now awaiting the reaction
    assert live.policy.level("visitor_1", "depth") == 1 and live.state()["pending"]["visitor_1"]["kind"] == "adapt"
    live.chat("no, that is too much, stop")                   # the very next message is the reaction
    assert live.policy.level("visitor_1", "depth") == 0
    rev = [r for r in live.policy.step_log if r["reversal"]]
    assert len(rev) == 1 and rev[0]["outcome"] == "declined"
    live.end()


def test_low_track_conf_drops_evidence_and_ui_state_shows_count():
    live = make("adult"); meet(live)
    live.profile.set_track_conf("visitor_1", 0.1)
    live.chat("The full story please")
    assert live.state()["suspended_writes"] >= 1 and live.state()["persons"]["visitor_1"]["dimensions"]["depth"]["evidence_count"] == 0
    live.end()


def test_end_wipes_and_refuses_disk_during_visit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    live = make("adult", log_dir=str(tmp_path / "logs")); meet(live)
    with pytest.raises(EphemeralViolation):
        SessionStore("sessions.db")
    live.chat("The full story please")
    live.end()
    assert live.state()["ended"] and live.state()["persons"] == {} and live.messages == []
    assert not ephemeral.active()
    files = [os.path.join(r, f) for r, _, fs in os.walk(tmp_path) for f in fs]
    assert files and all(f.endswith(".jsonl") for f in files)
    raw = open(files[0]).read().lower()
    assert "full story" not in raw and "visitor_1" in raw


def test_flask_endpoints():
    live = make("adult"); meet(live)
    c = L.make_app(live).test_client()
    assert c.get("/").status_code == 200 and b"Museum live profile" in c.get("/").data
    assert c.post("/chat", json={"text": "short version please"}).get_json()["reply"]
    s = c.get("/state").get_json()
    assert s["persons"]["visitor_1"]["dimensions"]["depth"]["value"] < 0
    assert c.post("/end").get_json()["persons_deleted"] == 1
    assert c.get("/state").get_json()["ended"]
    assert c.post("/new").get_json()["visit_id"]
    c.post("/end")


def test_no_face_yet_does_not_crash():
    live = make("adult")
    r = live.chat("hello")
    assert "can't see you" in r["reply"]
    live.end()


def test_interpret_choice_and_style_prompt():
    assert L.interpret_choice("depth", "the short version") == -1 and L.interpret_choice("depth", "full story") == 1
    assert L.interpret_choice("pace", "whatever") == 0
    p = L.style_instructions({"depth": -1, "talk_level": -1, "pace": 0}, True)
    assert "ONE short sentence" in p and "minimal" in p and "simple" in p and "mood" not in p.lower()


def test_one_frame_ghost_never_becomes_a_visitor_and_does_not_inflate_group_size():
    real = (100, 60, 240, 220)
    state = {"n": 0}
    emb = np.random.RandomState(7).randn(512); emb /= np.linalg.norm(emb)
    ghost_emb = np.random.RandomState(8).randn(512); ghost_emb /= np.linalg.norm(ghost_emb)
    def det(f):
        state["n"] += 1
        out = [Detection(real, emb)]
        if state["n"] == 2:
            out.append(Detection((400, 10, 440, 50), ghost_emb))        # appears in a single frame
        return out
    live = L.Live(None, lambda s, u: "{}", FakeAgeEstimator({"visitor_1": AgeProbs(0.06, 0.92, 0.02), "visitor_2": AgeProbs(0.5, 0.5, 0.0)}),
                  det, reply_fn=lambda s, t: "ok")
    for _ in range(8):
        live.process_frame(None)
    assert live.profile.persons == ["visitor_1"] and list(live.met) == ["visitor_1"]
    assert "solo" in live.met["visitor_1"].row_id            # group_size 1, not 2
    live.end()
