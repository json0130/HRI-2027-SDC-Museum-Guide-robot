import json

import pytest

from .config import load_config
from .profile import VisitProfile
from .profiling_policy import DEFAULTS, ProfilingPolicy


def make():
    prof = VisitProfile("v")
    prof.add_person("visitor_1")
    return prof, ProfilingPolicy(prof)


def test_gate_thresholds():
    _, pol = make()
    assert [pol.gate(c) for c in (0.0, 0.34, 0.35, 0.69, 0.7, 1.0)] == ["generic", "generic", "offer", "offer", "adapt", "adapt"]


def test_cold_start_choice_questions_first_then_one_open_then_budget_stops():
    prof, pol = make()
    kinds, dims = [], []
    for turn in range(0, 40):
        m = pol.next_move("visitor_1", turn)
        if m.kind != "none":
            kinds.append(m.kind); dims.append(m.dim)
            # questions are left unanswered on purpose: they expire after ignore_after_turns
    assert kinds == ["ask_choice", "ask_choice", "ask_choice", "ask_open"]
    assert dims[:3] == ["talk_level", "depth", "pace"]       # usefulness order, each dim once


def test_budget_min_turns_between_questions():
    _, pol = make()
    assert pol.next_move("visitor_1", 0).kind == "ask_choice"
    assert [pol.next_move("visitor_1", t).kind for t in (1, 2)] == ["none", "none"]
    assert pol.next_move("visitor_1", 3).kind == "ask_choice"


def test_all_text_comes_from_allow_list():
    cfg = load_config("museum_policy", DEFAULTS)
    allowed = {cfg["questions"]["open"], *cfg["questions"]["choice"].values(),
               *(t for d in cfg["questions"]["offer"].values() for t in d.values())}
    prof, pol = make()
    for t in range(60):
        m = pol.next_move("visitor_1", t)
        if m.text:
            assert m.text in allowed
        if t == 5:
            prof.record_style("visitor_1", "depth", +1, "llm_statement")
            prof.record_style("visitor_1", "depth", +1, "behavior_cue")


def test_low_confidence_keeps_generic_no_adapt():
    prof, pol = make()
    prof.record_style("visitor_1", "pace", +1, "behavior_cue")          # conf 0.25 < low
    assert all(pol.next_move("visitor_1", t).kind != "adapt" for t in range(6))
    assert pol.level("visitor_1", "pace") == 0


def test_medium_confidence_offers_then_accept_leads_to_one_step_adapt():
    prof = VisitProfile("v")
    prof.add_person("visitor_1")
    cfg = load_config("museum_policy", DEFAULTS)
    cfg["usefulness"] = {**cfg["usefulness"], "talk_level": 0.0, "pace": 0.0}   # make depth the argmax
    pol = ProfilingPolicy(prof, cfg)
    prof.record_style("visitor_1", "depth", +1, "llm_statement")
    prof.record_style("visitor_1", "depth", +1, "behavior_cue")          # conf 0.625 -> offer band
    m = pol.next_move("visitor_1", 0)
    assert (m.kind, m.dim, m.direction, m.text) == ("offer", "depth", 1, "Want more detail on this one?")
    pol.observe("visitor_1", "accepted", 1)                              # explicit yes = choice_answer evidence
    assert prof.style("visitor_1", "depth")["confidence"] > 0.7
    m = pol.next_move("visitor_1", 2)
    assert (m.kind, m.dim, m.level) == ("adapt", "depth", 1)             # ONE level, not two
    assert pol.level("visitor_1", "depth") == 1


def test_adapt_is_stepwise_with_cooldown_and_cap():
    prof, pol = make()
    prof.record_style("visitor_1", "talk_level", -1, "choice_answer")
    prof.record_style("visitor_1", "talk_level", -1, "llm_statement")    # high confidence, quiet
    levels = []
    for t in range(0, 12):
        m = pol.next_move("visitor_1", t)
        if m.kind == "adapt":
            levels.append(m.level)
            pol.observe("visitor_1", "accepted", t)
    assert levels == [-1, -2]                                             # one at a time, then capped at max_level


@pytest.mark.parametrize("outcome", ["declined", "ignored"])
def test_negative_outcome_reverts_step_lowers_confidence_and_logs(outcome):
    prof, pol = make()
    prof.record_style("visitor_1", "pace", +1, "choice_answer")
    prof.record_style("visitor_1", "pace", +1, "llm_statement")
    conf0 = prof.style("visitor_1", "pace")["confidence"]
    assert pol.next_move("visitor_1", 0).kind == "adapt" and pol.level("visitor_1", "pace") == 1
    if outcome == "declined":
        pol.observe("visitor_1", "declined", 1)
    else:
        pol.next_move("visitor_1", 5)                                     # unanswered -> expires as ignored
    assert pol.level("visitor_1", "pace") == 0
    assert prof.style("visitor_1", "pace")["confidence"] < conf0 * 0.6
    rev = [r for r in pol.step_log if r["reversal"]]
    assert len(rev) == 1 and rev[0]["outcome"] == outcome and (rev[0]["level_before"], rev[0]["level_after"]) == (1, 0)


def test_contradicting_evidence_after_adapt_reverts_automatically():
    prof, pol = make()
    prof.record_style("visitor_1", "talk_level", +1, "choice_answer")
    prof.record_style("visitor_1", "talk_level", +1, "llm_statement")
    assert pol.next_move("visitor_1", 0).kind == "adapt"
    pol.observe("visitor_1", "accepted", 1)
    for _ in range(4):
        prof.record_style("visitor_1", "talk_level", -1, "choice_answer")  # visitor now says the opposite
    pol.next_move("visitor_1", 2)
    assert pol.level("visitor_1", "talk_level") == 0
    assert [r["outcome"] for r in pol.step_log if r["reversal"]] == ["contradicted"]


def test_mixed_evidence_does_not_trigger_offer_or_adapt():
    prof, pol = make()
    for s in (+1, -1):
        prof.record_style("visitor_1", "depth", s, "choice_answer")      # value 0, conf high
    moves = [pol.next_move("visitor_1", t) for t in range(0, 20, 4)]
    assert not [m for m in moves if m.dim == "depth" and m.kind in ("adapt", "offer")]


def test_step_log_is_json_serialisable_and_has_no_prior():
    prof, pol = make()
    prof.set_prior("visitor_1", "child", 0.9)
    for t in range(8):
        pol.next_move("visitor_1", t)
    blob = json.dumps(pol.step_log)
    assert "child" not in blob and pol.step_log


def test_suspended_person_gets_no_moves_and_observe_writes_are_dropped():
    prof, pol = make()
    prof.record_style("visitor_1", "pace", +1, "choice_answer")
    prof.record_style("visitor_1", "pace", +1, "llm_statement")
    prof.set_track_conf("visitor_1", 0.1)
    assert pol.next_move("visitor_1", 0).kind == "none"
    assert pol.level("visitor_1", "pace") == 0
    prof.set_track_conf("visitor_1", 0.9)
    assert pol.next_move("visitor_1", 1).kind == "adapt"
    prof.set_track_conf("visitor_1", 0.1)
    n = prof.style("visitor_1", "pace")["evidence_count"]
    pol.observe("visitor_1", "declined", 2)                 # the revert still happens; the evidence write is dropped
    assert prof.style("visitor_1", "pace")["evidence_count"] == n and prof.suspended_writes >= 1
    assert pol.level("visitor_1", "pace") == 0
