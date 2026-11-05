"""No-hardware simulator: scripted visitors with ground truth, three modes, paired seeds.

    python -m modules.museum.simulate --seeds 200 --out docs/results --prefix museum_first_run

Modes: fixed (neutral opening, no profiling), prior-only (opening from the age-band prior, no
profiling), profile (prior + guarded extraction + profiling policy with small steps/reversals).
Everything is deterministic given the seed; every random draw is keyed by (seed, scenario, person,
turn, purpose), NOT by position in a stream, so a given visitor says and does the same things in
every mode (paired design). Visitor behaviour and the fake LLM are my assumptions
(config/museum_sim.yaml): the numbers measure the policy against these simulated people only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import initiation, profile as profile_mod, profiling_policy, appearance
from .appearance import AgeBandPipeline, AgeProbs, Detection, FakeAgeEstimator
from .config import load_config
from .decision_log import MODES, DecisionLog, snapshot
from .initiation import choose_opening
from .profile import DIMS, VisitProfile
from .profile_extraction import extract_and_apply_profile
from .profiling_policy import ProfilingPolicy

SIM_DEFAULTS = {
    "turns": 24, "seeds": 200, "boot": 1000, "noise": 0.1, "misread": 0.1,
    "noise_grid": [0.0, 0.1, 0.2, 0.3], "misread_grid": [0.0, 0.1, 0.2, 0.3], "prior_frames": 8,
    "seconds_per_turn": 20, "swap": {"start": 8, "window": 4}, "topic_call": 0.2, "style_call": 0.05,
    "neutral_fp": 0.2, "costs": {"question": [0.0, 0.05, 0.1]},
    "visitor": {"p_speak": {"-1": 0.25, "0": 0.5, "1": 0.8}, "p_answer": 0.9, "p_lapse": 0.1,
                "p_accept_agree": 0.9, "p_decline_oppose": 0.9, "p_accept_neutral": 0.5, "p_react_agree": 0.9,
                "p_react_oppose": 0.7, "p_decline_neutral": 0.2, "p_followup": 0.3},
}

STYLE_PHRASE = {("depth", -1): "just give me the short version", ("depth", 1): "tell me the full story please",
                ("pace", 1): "let's keep things moving", ("pace", -1): "can we take this slowly",
                ("talk_level", -1): "i'd rather you stayed quiet", ("talk_level", 1): "talking with you as we go suits me"}


@dataclass
class Visitor:
    band: str                                   # true age band: child | adult
    truth: Dict[str, int]                       # dim -> -1 | 0 | +1
    likes: List[tuple] = field(default_factory=list)      # (label, category)
    dislikes: List[tuple] = field(default_factory=list)
    force_misread: bool = False


NEUTRAL = {"talk_level": 0, "depth": 0, "pace": 0}
QUIET = Visitor("adult", {"talk_level": -1, "depth": 0, "pace": -1}, likes=[("old maps", "media")])
CHATTY = Visitor("child", {"talk_level": 1, "depth": 1, "pace": 1}, likes=[("dinosaurs", "animals")])
# name -> (swap kind, visitors)
SCENARIOS: Dict[str, tuple] = {
    "child_correct": (None, [Visitor("child", {"talk_level": 0, "depth": -1, "pace": 1}, likes=[("space rockets", "science")])]),
    "adult_correct": (None, [Visitor("adult", {"talk_level": 0, "depth": 1, "pace": 0}, likes=[("sculpture", "media")])]),
    "adult_misread_child": (None, [Visitor("adult", {"talk_level": 0, "depth": 1, "pace": 0}, likes=[("sculpture", "media")], force_misread=True)]),
    "quiet": (None, [QUIET]),
    "chatty": (None, [CHATTY]),
    "topic_dislike": (None, [Visitor("adult", dict(NEUTRAL), likes=[("sailing ships", "place")], dislikes=[("war", "other")])]),
    "swap_detected": ("detected", [QUIET, CHATTY]),
    "swap_silent": ("silent", [QUIET, CHATTY]),
    "swap_silent_persistent": ("persistent", [QUIET, CHATTY]),
}
SINGLE = [k for k, v in SCENARIOS.items() if v[0] is None]
SWAP = [k for k, v in SCENARIOS.items() if v[0] is not None]


def sim_cfg() -> dict:
    return load_config("museum_sim", SIM_DEFAULTS)


def effective_configs() -> dict:
    return {"appearance": load_config("museum_appearance", appearance.DEFAULTS),
            "initiation": load_config("museum_initiation", initiation.DEFAULTS),
            "profile": load_config("museum_profile", profile_mod.DEFAULTS),
            "policy": load_config("museum_policy", profiling_policy.DEFAULTS),
            "sim": sim_cfg()}


def config_hash() -> str:
    return hashlib.sha256(json.dumps(effective_configs(), sort_keys=True).encode()).hexdigest()


def u(*key) -> float:
    return random.Random("|".join(map(str, key))).random()


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


# ── fake LLM with a noise knob ───────────────────────────────────────────────────────────

def make_fake_llm(noise: float, key: tuple):
    """Reads the templated sentences. Per extracted item, with probability ``noise`` it is corrupted in one
    of: flip the sign/direction (guards cannot catch), invent a quote (guard drops), low confidence (guard drops),
    off-vocabulary category/dim (guard drops). With probability noise/2 it also adds a spurious topic quoting a
    real word of the utterance (guards cannot catch)."""
    def llm(system: str, text: str) -> str:
        low, topics, style = text.lower(), [], []
        for (dim, d), phrase in STYLE_PHRASE.items():
            if phrase in low:
                style.append({"dim": dim, "direction": "high" if d > 0 else "low", "confidence": 0.9, "quote": phrase})
        for kind, sign in (("i love ", "like"), ("i can't stand ", "dislike")):
            if kind in low:
                label = low.split(kind, 1)[1].strip(" .!")
                topics.append({"label": label, "category": "other", "sign": sign, "confidence": 0.9, "quote": kind + label})
        for i, it in enumerate(style + topics):
            if u(*key, "noise", i) < noise:
                how = int(u(*key, "how", i) * 4)
                if how == 0:
                    if "direction" in it: it["direction"] = "low" if it["direction"] == "high" else "high"
                    else: it["sign"] = "dislike" if it["sign"] == "like" else "like"
                elif how == 1: it["quote"] = "something never said"
                elif how == 2: it["confidence"] = 0.3
                else:
                    if "dim" in it: it["dim"] = "mood"
                    else: it["category"] = "weird"
        if u(*key, "spurious") < noise / 2:
            words = [w for w in low.split() if len(w) > 3]
            if words:
                topics.append({"label": words[0], "category": "other", "sign": "like", "confidence": 0.9, "quote": words[0]})
        return json.dumps({"topics": topics, "style": style})
    return llm


# ── one visit ────────────────────────────────────────────────────────────────────────────

def _empty() -> dict:
    keys = ("persons", "open_right", "open_wrong", "adapt", "adapt_right", "wrong_opposed", "wrong_neutral", "reversals",
            "questions", "offers_declined", "suspended", "conf_dims", "conf_reached", "conf_turns", "conf_wrong",
            "acc_dims", "acc_correct", "acc_wrong", "acc_none", "neutral_dims", "neutral_fp", "mae_sum", "mae_n",
            "like_tp", "like_fp", "like_fn", "dis_tp", "dis_fp", "dis_fn")
    return {k: 0.0 for k in keys}


def run_visit(scenario: str, mode: str, seed: int, *, noise: float, misread: float, cfg: Optional[dict] = None,
              log_dir: Optional[str] = None) -> dict:
    cfg = cfg or sim_cfg()
    v = cfg["visitor"]
    swap_kind, vis = SCENARIOS[scenario]
    n, T = len(vis), cfg["turns"]
    ids = [f"visitor_{i + 1}" for i in range(n)]
    m = _empty()
    m["persons"] = n
    prof = VisitProfile(f"sim-{scenario}-{seed}")
    for pid in ids:
        prof.add_person(pid)
    pol = ProfilingPolicy(prof)
    state = {"t": 0}
    log = DecisionLog(log_dir, mode, clock=lambda: state["t"] * cfg["seconds_per_turn"]) if log_dir else None
    try:
        # opening: camera estimate -> (maybe) prior -> opening row
        script, dets = {}, []
        for i, vis_i in enumerate(vis):
            wrong = vis_i.force_misread or u(seed, scenario, i, "misread") < misread
            read = ("adult" if vis_i.band == "child" else "child") if wrong else vis_i.band
            script[ids[i]] = AgeProbs(0.88, 0.10, 0.02) if read == "child" else AgeProbs(0.07, 0.91, 0.02)
            e = np.random.RandomState(seed * 31 + i).randn(512)
            dets.append(Detection((100 * i, 0, 100 * i + 40, 40), e / np.linalg.norm(e)))
        pipe = AgeBandPipeline(FakeAgeEstimator(script, noise=0.05, seed=seed), lambda f: dets)
        for _ in range(cfg["prior_frames"]):
            res = pipe.process(None)
        pipe.clear()
        prior_log: Dict[str, Optional[dict]] = {}
        for i, r in enumerate(res):
            use_prior = mode != "fixed"
            op = choose_opening(r["band"], r["confidence"], n, use_prior=use_prior)
            prior_log[ids[i]] = {"band": r["band"], "confidence": r["confidence"]} if use_prior else None
            if use_prior:
                prof.set_prior(ids[i], r["band"], r["confidence"])
            cls = op.row_id.split("_")[0]
            if cls in ("child", "adult"):
                m["open_right" if cls == vis[i].band else "open_wrong"] += 1
        # turns
        confident: Dict[tuple, Optional[int]] = {}
        for t in range(T):
            state["t"] = t
            in_win = swap_kind and cfg["swap"]["start"] <= t < cfg["swap"]["start"] + (10**6 if swap_kind == "persistent" else cfg["swap"]["window"])
            swapped = bool(in_win) and n == 2
            phys_of = {ids[0]: 1, ids[1]: 0} if swapped else {pid: i for i, pid in enumerate(ids)}
            if mode == "profile":
                for pid in ids:
                    prof.set_track_conf(pid, 0.2 if (swap_kind == "detected" and in_win) else 0.9)
            for pid in ids:
                ph = phys_of[pid]
                tr = vis[ph].truth
                key = (seed, scenario, ph, t)
                move, outcome, reversal0 = None, "none", len([r for r in pol.step_log if r["reversal"]])
                if mode == "profile":
                    move = pol.next_move(pid, t)
                    if move.kind == "adapt":
                        m["adapt"] += 1
                        d, dr = move.dim, move.direction
                        if tr[d] == dr: m["adapt_right"] += 1
                        elif tr[d] == -dr: m["wrong_opposed"] += 1
                        else: m["wrong_neutral"] += 1
                        x = u(*key, "adapt")
                        if tr[d] == dr: outcome = "accepted" if x < v["p_react_agree"] else "none"
                        elif tr[d] == -dr: outcome = "declined" if x < v["p_react_oppose"] else "none"
                        else: outcome = "declined" if x < v["p_decline_neutral"] else "accepted"
                        if outcome != "none": pol.observe(pid, outcome, t)
                    elif move.kind == "offer":
                        m["questions"] += 1
                        d, dr, x = move.dim, move.direction, u(*key, "offer")
                        if tr[d] == dr: outcome = "accepted" if x < v["p_accept_agree"] else "declined"
                        elif tr[d] == -dr: outcome = "declined" if x < v["p_decline_oppose"] else "accepted"
                        else: outcome = "accepted" if x < v["p_accept_neutral"] else "declined"
                        m["offers_declined"] += outcome == "declined"
                        pol.observe(pid, outcome, t)
                    elif move.kind == "ask_choice":
                        m["questions"] += 1
                        d = move.dim
                        if tr[d] != 0 and u(*key, "answer") < v["p_answer"]:
                            sign = tr[d] * (-1 if u(*key, "lapse") < v["p_lapse"] else 1)
                            pol.answer_choice(pid, d, sign, t)
                            outcome = "answered"
                    elif move.kind == "ask_open":
                        m["questions"] += 1
                    # spontaneous statement
                    cands = [("style", d, tr[d]) for d in DIMS if tr[d]] + [("like", *x) for x in vis[ph].likes] + \
                            [("dislike", *x) for x in vis[ph].dislikes]
                    forced = move is not None and move.kind == "ask_open" and vis[ph].likes and u(*key, "answer") < v["p_answer"]
                    if cands and (forced or u(*key, "speak") < v["p_speak"][str(tr["talk_level"])]):
                        pool = [("like", *x) for x in vis[ph].likes] if forced else cands
                        c = pool[int(u(*key, "pick") * len(pool))]
                        text = STYLE_PHRASE[(c[1], c[2])] if c[0] == "style" else \
                            (f"i love {c[1]}" if c[0] == "like" else f"i can't stand {c[1]}")
                        extract_and_apply_profile(prof, pid, text.capitalize() + ".", make_fake_llm(noise, key), t)
                    if tr["depth"] == 1 and u(*key, "followup") < v["p_followup"]:
                        prof.record_style(pid, "depth", 1, "behavior_cue", t)
                if log:
                    new_rev = len([r for r in pol.step_log if r["reversal"]]) > reversal0
                    gate = pol.gate(prof.style(pid, move.dim)["confidence"]) if (move and move.dim) else "none"
                    log.log(pid, prior=prior_log[pid], profile=snapshot(prof, pid),
                            action={"kind": move.kind if move else "none", "dim": move.dim if move else None,
                                    "direction": move.direction if move else 0, "level": move.level if move else None},
                            gate=gate, outcome=outcome, reversal=new_rev, track_conf=prof._track_conf.get(pid, 1.0),
                            suspended_writes=prof.suspended_writes)
            if mode == "profile":
                for i, pid in enumerate(ids):                         # turns-to-confident against the NOMINAL person
                    for d in DIMS:
                        if vis[i].truth[d] and (pid, d) not in confident:
                            st = prof.style(pid, d)
                            if st["confidence"] >= pol.cfg["thresholds"]["high"]:
                                confident[(pid, d)] = t + 1
                                m["conf_reached"] += 1
                                m["conf_turns"] += t + 1
                                m["conf_wrong"] += _sign(st["value"]) != vis[i].truth[d]
        if mode == "profile":
            for i, pid in enumerate(ids):
                for d in DIMS:
                    st, tv = prof.style(pid, d), vis[i].truth[d]
                    if tv:
                        m["conf_dims"] += 1; m["acc_dims"] += 1
                        s = _sign(st["effective_value"]) if abs(st["effective_value"]) >= cfg["style_call"] else 0
                        if s == tv: m["acc_correct"] += 1
                        elif s == 0: m["acc_none"] += 1
                        else: m["acc_wrong"] += 1
                    else:
                        m["neutral_dims"] += 1; m["neutral_fp"] += abs(st["effective_value"]) >= cfg["neutral_fp"]
                    m["mae_sum"] += abs(st["effective_value"] - tv); m["mae_n"] += 1
                tp = prof.topics(pid)
                for kind, truth_list in (("like", vis[i].likes), ("dis", vis[i].dislikes)):
                    pred = {r["label"].lower() for r in tp["likes" if kind == "like" else "dislikes"] if abs(r["effective_value"]) >= cfg["topic_call"]}
                    real = {x[0].lower() for x in truth_list}
                    m[f"{kind}_tp"] += len(pred & real); m[f"{kind}_fp"] += len(pred - real); m[f"{kind}_fn"] += len(real - pred)
            m["reversals"] = len([r for r in pol.step_log if r["reversal"]])
            m["suspended"] = prof.suspended_writes
    finally:
        if log: log.close()
        prof.end()
    return m


# ── aggregation: paired seeds, bootstrap over seeds ──────────────────────────────────────

def run_grid(scenarios, modes, seeds, *, noise, misread, cfg, log_dir=None) -> Dict[str, Dict[str, np.ndarray]]:
    """-> {mode: {metric: array over seeds (summed over the scenarios' persons)}}"""
    out = {md: {k: np.zeros(len(seeds)) for k in _empty()} for md in modes}
    for si, seed in enumerate(seeds):
        for md in modes:
            for sc in scenarios:
                r = run_visit(sc, md, seed, noise=noise, misread=misread, cfg=cfg,
                              log_dir=log_dir if (seed == seeds[0]) else None)
                for k, x in r.items():
                    out[md][k][si] += x
    return out


def boot_ci(num: np.ndarray, den: Optional[np.ndarray], B: int, rng: np.random.RandomState):
    """Point estimate and 95% percentile bootstrap interval of sum(num)/sum(den) (or mean num) over seeds."""
    S = len(num)
    idx = rng.randint(0, S, size=(B, S))
    if den is None:
        f = lambda a: a.mean(); est = num.mean(); bs = num[idx].mean(axis=1)
    else:
        est = num.sum() / den.sum() if den.sum() else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            bs = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    bs = bs[~np.isnan(bs)]
    return est, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) else (float("nan"),) * 2


def summarize(g: Dict[str, np.ndarray], B: int, seed: int = 12345) -> Dict[str, tuple]:
    rng = np.random.RandomState(seed)
    P = g["persons"]
    per = lambda k: boot_ci(g[k], P, B, rng)
    right = g["adapt_right"] + g["open_right"]
    wrong = g["wrong_opposed"] + g["wrong_neutral"] + g["open_wrong"]
    rows = {
        "wrong_openings": per("open_wrong"), "right_openings": per("open_right"),
        "adaptations": per("adapt"), "right_adaptations": per("adapt_right"),
        "wrong_adapt_opposed": per("wrong_opposed"), "wrong_adapt_neutral": per("wrong_neutral"),
        "reversals": per("reversals"), "questions": per("questions"), "offers_declined": per("offers_declined"),
        "suspended_writes": per("suspended"),
        "turns_to_confident": boot_ci(g["conf_turns"], g["conf_reached"], B, rng),
        "confident_dims_frac": boot_ci(g["conf_reached"], g["conf_dims"], B, rng),
        "confident_but_wrong": boot_ci(g["conf_wrong"], g["conf_reached"], B, rng),
        "acc_correct": boot_ci(g["acc_correct"], g["acc_dims"], B, rng), "acc_wrong": boot_ci(g["acc_wrong"], g["acc_dims"], B, rng),
        "acc_no_call": boot_ci(g["acc_none"], g["acc_dims"], B, rng),
        "neutral_false_pos": boot_ci(g["neutral_fp"], g["neutral_dims"], B, rng),
        "mae_eff_value": boot_ci(g["mae_sum"], g["mae_n"], B, rng),
        "like_precision": boot_ci(g["like_tp"], g["like_tp"] + g["like_fp"], B, rng),
        "like_recall": boot_ci(g["like_tp"], g["like_tp"] + g["like_fn"], B, rng),
        "dislike_precision": boot_ci(g["dis_tp"], g["dis_tp"] + g["dis_fp"], B, rng),
        "dislike_recall": boot_ci(g["dis_tp"], g["dis_tp"] + g["dis_fn"], B, rng),
    }
    return rows


def break_even(g: Dict[str, np.ndarray], q_cost: float, B: int, seed: int = 777):
    """SENSITIVITY ANALYSIS, not a measured cost: wrong/right cost ratio W/R at which a mode's net benefit
    (R*right - W*wrong - q*questions, right/wrong counting non-neutral openings and adaptation steps) equals
    fixed's (0, since fixed never adapts). Above it, `fixed` beats the mode."""
    right = g["adapt_right"] + g["open_right"] - q_cost * g["questions"]
    wrong = g["wrong_opposed"] + g["wrong_neutral"] + g["open_wrong"]
    return boot_ci(right, wrong, B, np.random.RandomState(seed))


def net_vs_fixed(res, mode, ratio, q_cost, B, seed=99):
    """Paired per-seed net benefit of `mode` minus `fixed` at wrong/right cost ratio `ratio`."""
    def net(g): return g["adapt_right"] + g["open_right"] - ratio * (g["wrong_opposed"] + g["wrong_neutral"] + g["open_wrong"]) - q_cost * g["questions"]
    d = net(res[mode]) - net(res["fixed"])
    return boot_ci(d / np.maximum(res[mode]["persons"], 1), None, B, np.random.RandomState(seed))


def git_info() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=here, text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


def fmt(t, p=2) -> str:
    est, (lo, hi) = t
    if est != est:
        return "n/a"
    return f"{est:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def full_run(seeds: List[int], cfg: dict, out_dir: Optional[str], prefix: str, log_dir: Optional[str], sweeps: bool = True) -> dict:
    B = cfg["boot"]
    base = {"config_hash": config_hash(), "git": git_info(), "seeds": [seeds[0], seeds[-1], len(seeds)],
            "headline_noise": cfg["noise"], "headline_misread": cfg["misread"], "turns": cfg["turns"], "modes": list(MODES)}
    result = {"meta": base, "headline": {}, "per_scenario": {}, "swap": {}, "sweeps": {"noise": {}, "misread": {}}, "break_even": {}, "net_vs_fixed": {}}
    head = run_grid(SINGLE, MODES, seeds, noise=cfg["noise"], misread=cfg["misread"], cfg=cfg, log_dir=log_dir)
    result["headline"] = {md: summarize(head[md], B) for md in MODES}
    for sc in SINGLE:
        g = run_grid([sc], MODES, seeds, noise=cfg["noise"], misread=cfg["misread"], cfg=cfg)
        result["per_scenario"][sc] = {md: summarize(g[md], B) for md in MODES}
    for sc in SWAP:
        g = run_grid([sc], MODES, seeds, noise=cfg["noise"], misread=cfg["misread"], cfg=cfg)
        result["swap"][sc] = {md: summarize(g[md], B) for md in MODES}
    swap_all = run_grid(SWAP, MODES, seeds, noise=cfg["noise"], misread=cfg["misread"], cfg=cfg)
    for q in cfg["costs"]["question"]:
        result["break_even"][str(q)] = {md: break_even(head[md], q, B) for md in ("prior-only", "profile")}
        result["break_even"][str(q)]["profile_with_swap_scenarios"] = break_even(
            {k: head["profile"][k] + swap_all["profile"][k] for k in head["profile"]}, q, B)
    for ratio in (1, 2, 5):
        result["net_vs_fixed"][str(ratio)] = {md: net_vs_fixed(head, md, ratio, 0.0, B) for md in ("prior-only", "profile")}
    if sweeps:
        for nz in cfg["noise_grid"]:
            g = run_grid(SINGLE, ("profile",), seeds, noise=nz, misread=cfg["misread"], cfg=cfg)
            result["sweeps"]["noise"][str(nz)] = summarize(g["profile"], B)
        for mr in cfg["misread_grid"]:
            g = run_grid(SINGLE, MODES, seeds, noise=cfg["noise"], misread=mr, cfg=cfg)
            result["sweeps"]["misread"][str(mr)] = {md: summarize(g[md], B) for md in MODES}
        result["break_even_by_misread"] = {}
        for mr in cfg["misread_grid"]:
            g = run_grid(SINGLE, ("fixed", "prior-only", "profile"), seeds, noise=cfg["noise"], misread=mr, cfg=cfg)
            result["break_even_by_misread"][str(mr)] = {md: break_even(g[md], 0.0, B) for md in ("prior-only", "profile")}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, prefix + ".json"), "w") as f:
            json.dump(result, f, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
        with open(os.path.join(out_dir, prefix + ".md"), "w") as f:
            f.write(render_md(result))
    return result


def render_md(r: dict) -> str:
    m = r["meta"]
    L = [f"# Museum simulator results ({m['seeds'][2]} paired seeds {m['seeds'][0]}..{m['seeds'][1]})", "",
         f"- config hash (sha256 of all effective museum configs): `{m['config_hash']}`",
         f"- git: `{m['git']['commit']}` dirty={m['git']['dirty']}",
         f"- headline noise={m['headline_noise']}, misread={m['headline_misread']}, turns={m['turns']}",
         "- Simulated visitors and fake LLM are my assumptions (config/museum_sim.yaml); numbers are per visitor, mean [95% bootstrap CI over seeds].", ""]
    keys = ["wrong_openings", "right_openings", "adaptations", "right_adaptations", "wrong_adapt_opposed", "wrong_adapt_neutral", "reversals",
            "questions", "offers_declined", "suspended_writes", "turns_to_confident", "confident_dims_frac", "confident_but_wrong",
            "acc_correct", "acc_wrong", "acc_no_call", "neutral_false_pos", "mae_eff_value", "like_precision", "like_recall", "dislike_precision", "dislike_recall"]
    def table(title, d, extra=""):
        out = [f"## {title}", "", "| metric | " + " | ".join(d) + " |", "|---|" + "---|" * len(d)]
        for k in keys:
            out.append(f"| {k} | " + " | ".join(fmt(d[md][k]) for md in d) + " |")
        return out + [""]
    L += table("Headline: single-visitor scenarios, all six", r["headline"])
    for sc, d in r["per_scenario"].items():
        L += table(f"Scenario: {sc}", d)
    for sc, d in r["swap"].items():
        L += table(f"Swap scenario: {sc}", d)
    L += ["## Break-even (SENSITIVITY ANALYSIS, not a measured cost)", "",
          "W/R at which the mode's net benefit equals `fixed` (0). `right` = right adaptation steps + right non-neutral openings, `wrong` = opposed + neutral-truth steps + wrong openings; "
          "q = cost per question in units of one right adaptation. Above this ratio `fixed` is better.", "",
          "| q | prior-only | profile | profile incl. swap scenarios |", "|---|---|---|---|"]
    for q, d in r["break_even"].items():
        L.append(f"| {q} | {fmt(d['prior-only'])} | {fmt(d['profile'])} | {fmt(d['profile_with_swap_scenarios'])} |")
    L += ["", "Net benefit per visitor vs fixed (paired seeds), at wrong/right cost ratio W/R, q=0:", "", "| W/R | prior-only | profile |", "|---|---|---|"]
    for ratio, d in r["net_vs_fixed"].items():
        L.append(f"| {ratio} | {fmt(d['prior-only'])} | {fmt(d['profile'])} |")
    if r["sweeps"]["noise"]:
        L += ["", "## Extraction-noise sweep (profile mode, misread at headline)", "", "| noise | " + " | ".join(["wrong_adapt_opposed", "wrong_adapt_neutral", "reversals", "turns_to_confident", "acc_correct", "acc_wrong", "confident_but_wrong"]) + " |", "|---|" + "---|" * 7]
        for nz, d in r["sweeps"]["noise"].items():
            L.append(f"| {nz} | " + " | ".join(fmt(d[k]) for k in ["wrong_adapt_opposed", "wrong_adapt_neutral", "reversals", "turns_to_confident", "acc_correct", "acc_wrong", "confident_but_wrong"]) + " |")
        L += ["", "## Age-misread sweep (noise at headline)", "", "| misread | prior-only wrong_openings | profile wrong_openings | profile wrong_adapt_opposed | break-even W/R prior-only | break-even W/R profile |", "|---|---|---|---|---|---|"]
        for mr, d in r["sweeps"]["misread"].items():
            be = r["break_even_by_misread"][mr]
            L.append(f"| {mr} | {fmt(d['prior-only']['wrong_openings'])} | {fmt(d['profile']['wrong_openings'])} | {fmt(d['profile']['wrong_adapt_opposed'])} | {fmt(be['prior-only'])} | {fmt(be['profile'])} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=None, help="number of paired seeds (default from config)")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="directory for <prefix>.json / .md")
    ap.add_argument("--prefix", default="museum_sim")
    ap.add_argument("--log-dir", default=None, help="write JSONL decision logs (first seed only)")
    ap.add_argument("--no-sweeps", action="store_true")
    ap.add_argument("--print-hash", action="store_true")
    a = ap.parse_args(argv)
    if a.print_hash:
        print(config_hash()); return 0
    cfg = sim_cfg()
    n = a.seeds or cfg["seeds"]
    res = full_run(list(range(a.first_seed, a.first_seed + n)), cfg, a.out, a.prefix, a.log_dir, sweeps=not a.no_sweeps)
    print(render_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
