"""Simulator tests. They use seeds >= 10000 and tiny runs so the official run (seeds 0..199) is never
used for development or tuning; they assert invariants, not that `profile` wins."""
import hashlib
import json
import os

import numpy as np
import pytest

from . import simulate as sim
from .decision_log import DecisionLog

CFG = sim.sim_cfg()
S = 10000


def run(sc, mode, seed=S, noise=0.0, misread=0.0):
    return sim.run_visit(sc, mode, seed, noise=noise, misread=misread, cfg=CFG)


def test_deterministic_and_seed_sensitive():
    assert run("chatty", "profile", noise=0.2) == run("chatty", "profile", noise=0.2)
    assert any(run("chatty", "profile", seed=S + i, noise=0.2) != run("chatty", "profile", seed=S, noise=0.2) for i in range(1, 6))


def test_fixed_and_prior_only_never_adapt_or_ask():
    for sc in sim.SCENARIOS:
        for md in ("fixed", "prior-only"):
            r = run(sc, md)
            assert r["adapt"] == r["questions"] == r["reversals"] == r["suspended"] == 0, (sc, md)
    assert run("quiet", "fixed")["open_right"] == run("quiet", "fixed")["open_wrong"] == 0     # neutral opening only


def test_openings_prior_uses_estimate_fixed_ignores_it():
    assert (run("adult_correct", "prior-only")["open_right"], run("adult_correct", "prior-only")["open_wrong"]) == (1, 0)
    r = run("adult_misread_child", "prior-only")                         # forced confident misread
    assert (r["open_right"], r["open_wrong"]) == (0, 1)
    assert run("adult_misread_child", "fixed")["open_wrong"] == 0
    # profile mode opens exactly like prior-only (paired)
    assert run("adult_misread_child", "profile")["open_wrong"] == 1


def test_misread_rate_is_a_real_parameter():
    n = sum(run("adult_correct", "prior-only", seed=S + i, misread=1.0)["open_wrong"] for i in range(20))
    z = sum(run("adult_correct", "prior-only", seed=S + i, misread=0.0)["open_wrong"] for i in range(20))
    assert (n, z) == (20, 0)
    mid = sum(run("adult_correct", "prior-only", seed=S + i, misread=0.5)["open_wrong"] for i in range(200))
    assert 70 < mid < 130


def test_paired_visitor_behaviour_same_utterances_across_modes():
    assert [sim.u(S, "x", 0, t, "speak") for t in range(5)] == [sim.u(S, "x", 0, t, "speak") for t in range(5)]


def test_noise_zero_produces_no_guard_evasion_and_noise_changes_outcomes():
    a = [run("chatty", "profile", seed=S + i, noise=0.0) for i in range(30)]
    b = [run("chatty", "profile", seed=S + i, noise=0.3) for i in range(30)]
    assert sum(x["acc_wrong"] for x in a) <= sum(x["acc_wrong"] for x in b)
    assert sum(x["acc_correct"] for x in a) >= sum(x["acc_correct"] for x in b)


def test_fake_llm_noise_events_are_caught_or_not_by_the_real_guards():
    from .profile import VisitProfile
    from .profile_extraction import extract_and_apply_profile
    text = "Talking with you as we go suits me. I love dinosaurs."
    wrong_sign = 0
    for i in range(300):
        p = VisitProfile("v"); p.add_person("visitor_1")
        extract_and_apply_profile(p, "visitor_1", text, sim.make_fake_llm(0.5, (S, i)), 0)
        wrong_sign += p.style("visitor_1", "talk_level")["value"] < 0
        p.end()
    assert 0 < wrong_sign < 300        # flips get through the guards; the other corruptions are dropped by them


def test_swap_detected_suspends_silent_does_not():
    d, s = run("swap_detected", "profile"), run("swap_silent", "profile")
    assert d["suspended"] > 0 and s["suspended"] == 0
    assert run("swap_detected", "fixed")["suspended"] == 0


def test_silent_swap_hurts_more_than_detected_over_seeds():
    sw = lambda sc: sum(run(sc, "profile", seed=S + i)["wrong_opposed"] + run(sc, "profile", seed=S + i)["acc_wrong"] for i in range(40))
    assert sw("swap_silent") > sw("swap_detected")


def test_wrong_adapt_split_opposed_vs_neutral():
    r = [run("topic_dislike", "profile", seed=S + i, noise=0.3) for i in range(40)]
    assert all(x["wrong_opposed"] == 0 for x in r)        # truth 0 on every style dim: nothing can oppose it
    assert sum(x["wrong_neutral"] for x in r) == sum(x["adapt"] for x in r)


def test_bootstrap_and_breakeven_arithmetic():
    rng = np.random.RandomState(0)
    est, (lo, hi) = sim.boot_ci(np.array([1.0, 2, 3, 4]), np.array([2.0, 2, 2, 2]), 500, rng)
    assert est == pytest.approx(1.25) and lo <= est <= hi
    g = {k: np.zeros(4) for k in sim._empty()}
    g["adapt_right"] += 3; g["open_right"] += 1; g["wrong_opposed"] += 1; g["wrong_neutral"] += 1; g["open_wrong"] += 0; g["questions"] += 2
    assert sim.break_even(g, 0.0, 200)[0] == pytest.approx(2.0)          # 4 right / 2 wrong
    assert sim.break_even(g, 0.5, 200)[0] == pytest.approx(1.5)          # (4 - 0.5*2) / 2


def test_config_hash_tracks_effective_config():
    h = sim.config_hash()
    assert h == hashlib.sha256(json.dumps(sim.effective_configs(), sort_keys=True).encode()).hexdigest()
    other = sim.effective_configs(); other["policy"]["thresholds"]["high"] = 0.71
    assert hashlib.sha256(json.dumps(other, sort_keys=True).encode()).hexdigest() != h


# ── decision log ─────────────────────────────────────────────────────────────────────────

def test_end_to_end_visit_in_empty_dir_creates_only_the_decision_log(tmp_path, monkeypatch):
    work = tmp_path / "empty"; work.mkdir(); monkeypatch.chdir(work)
    logs = work / "logs" / "museum"
    for md in sim.MODES:
        for sc in ("topic_dislike", "swap_detected"):
            sim.run_visit(sc, md, S, noise=0.2, misread=0.1, cfg=CFG, log_dir=str(logs))
    files = [os.path.join(r, f) for r, _, fs in os.walk(work) for f in fs]
    assert files and all(f.startswith(str(logs)) and f.endswith(".jsonl") for f in files), files
    assert len(files) == 6                                      # one log per visit, nothing else (no faces.npz, sessions.db, kg_state.json)


def test_log_content_is_anonymous_whitelisted_and_random_id(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    sim.run_visit("topic_dislike", "profile", S, noise=0.0, misread=0.0, cfg=CFG, log_dir=str(a))
    sim.run_visit("topic_dislike", "profile", S, noise=0.0, misread=0.0, cfg=CFG, log_dir=str(b))
    fa, fb = [json.loads(l) for l in open(next(a.iterdir()))], [json.loads(l) for l in open(next(b.iterdir()))]
    assert fa[0]["visit_id"] != fb[0]["visit_id"] and len(fa[0]["visit_id"]) == 12          # random, not seed-derived
    raw = open(next(a.iterdir())).read().lower()
    for leak in ("war", "sailing", "ships", "short version", "love", "stand"):               # topic labels / visitor speech
        assert leak not in raw
    assert set(fa[0]) == {"t", "visit_id", "mode", "person", "prior", "profile", "action", "gate", "outcome", "reversal",
                          "track_conf", "suspended_writes"}
    assert set(fa[0]["action"]) == {"kind", "dim", "direction", "level"}
    assert fa[0]["t"] == 0 and [r["t"] for r in fa] == sorted(r["t"] for r in fa) and fa[1]["t"] == CFG["seconds_per_turn"]
    assert all("topics_by_category" in r["profile"] for r in fa)


def test_decision_log_clock_and_mode_validation(tmp_path):
    clk = iter([100.0, 107.5])
    log = DecisionLog(str(tmp_path), "profile", clock=lambda: next(clk))
    log.log("visitor_1", prior=None, profile={}, action={"kind": "none"}, gate="none", outcome="none", reversal=False,
            track_conf=0.9, suspended_writes=0)
    log.close()
    row = json.loads(open(log.path).read())
    assert row["t"] == 7.5 and row["prior"] is None
    with pytest.raises(ValueError):
        DecisionLog(str(tmp_path), "bogus")
