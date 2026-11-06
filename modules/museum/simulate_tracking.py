"""No-hardware simulator for steps 1-4 ONLY: tracker identity, re-association, write suspension, group context and the
engagement trigger. (The older simulate.py evaluates the step 5/6 placeholders; it is frozen and not touched here.)

    python -m modules.museum.simulate_tracking --seeds 100 --out docs/results --prefix museum_tracking_first_run

Scripted people with ground-truth identity walk through a synthetic scene; per frame the REAL AgeBandPipeline (tracker),
GroupTracker, EngagementEngine and VisitProfile run on fake detections (box + a synthetic face embedding). This is MY model
of people and of face embeddings: synthetic embeddings show that the logic works, not that real faces separate (relatives
especially), and nothing here measures real visitors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import appearance, engagement, group, initiation, profile as profile_mod
from .appearance import AgeBandPipeline, AgeProbs, Detection
from .config import load_config
from .decision_log import DecisionLog
from .engagement import EngagementEngine, opening_for_event
from .group import GroupTracker
from .profile import VisitProfile
from .simulate import boot_ci

DEFAULTS = {"fps": 15, "frame_w": 640, "frame_h": 480, "seeds": 100, "boot": 1000, "genuine_cos": 0.8, "relative_cos": 0.8,
            "write_every_s": 1.0, "child_scale": 0.7, "pos_jitter_px": 3, "ratio_jitter": 0.03, "time_offset_s": 0.4}


def cfg() -> dict:
    return load_config("museum_trackeval", DEFAULTS)


def effective_configs() -> dict:
    return {"appearance": load_config("museum_appearance", appearance.DEFAULTS), "group": load_config("museum_group", group.DEFAULTS),
            "engagement": load_config("museum_engagement", engagement.DEFAULTS), "initiation": load_config("museum_initiation", initiation.DEFAULTS),
            "profile": load_config("museum_profile", profile_mod.DEFAULTS), "trackeval": cfg()}


def config_hash() -> str:
    return hashlib.sha256(json.dumps(effective_configs(), sort_keys=True).encode()).hexdigest()


# ── scripted people ──────────────────────────────────────────────────────────────────────

def path(*pts):
    """Piecewise-linear (t, cx, cy, ratio) waypoints; None before the first and after the last (or between two waypoints
    marked by a (t, None) entry: absent from that t until the next waypoint)."""
    def f(t):
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a[0] <= t <= b[0]:
                if a[1] is None:
                    return None
                u = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
                if b[1] is None:
                    return (a[1], a[2], a[3])
                return (a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * (b[3] - a[3]))
        return None
    return f


@dataclass
class Person:
    name: str
    role: str                    # "visitor" (intends to engage), "passer" (must NOT trigger), "far" (must NOT trigger)
    band: str                    # true age band
    traj: Callable               # t -> (cx, cy, ratio) or None
    sign: int = 1                # the sign of the evidence this person "says" (to detect misattribution)
    relative_of: Optional[str] = None
    scale: float = 1.0


def walk_across(y, ratio, t0, dur, left_to_right=True, name="p", role="passer", band="adult"):
    x0, x1 = (60, 580) if left_to_right else (580, 60)
    return Person(name, role, band, path((t0, x0, y, ratio), (t0 + dur, x1, y, ratio)))


def approach_and_stay(x, ratio, t_arrive, t_end, name="v", role="visitor", band="adult", sign=1, start_ratio=0.08, approach_s=3.0, y=220, scale=1.0, t_start=0.0):
    return Person(name, role, band, path((t_start, x, y, start_ratio * scale), (t_arrive, x, y, ratio * scale), (t_end, x, y, ratio * scale)), sign=sign, scale=scale)


def scenarios(c: dict) -> Dict[str, dict]:
    cs = c["child_scale"]
    S: Dict[str, dict] = {}
    S["crossing"] = {"emb": True, "T": 12, "people": [
        Person("A", "passer", "adult", path((0, 100, 220, .3), (10, 540, 220, .3)), sign=1),
        Person("B", "passer", "adult", path((0, 540, 220, .3), (10, 100, 220, .3)), sign=-1)]}
    S["crossing_iou_only"] = {**S["crossing"], "emb": False}
    ex = lambda: [Person("A", "visitor", "adult", path((0, 160, 220, .3), (5, 160, 220, .3), (5, None, 0, 0), (6, 480, 220, .3), (12, 480, 220, .3)), sign=1),
                  Person("B", "visitor", "adult", path((0, 480, 220, .3), (5, 480, 220, .3), (5, None, 0, 0), (6, 160, 220, .3), (12, 160, 220, .3)), sign=-1)]
    S["exchange_places"] = {"emb": True, "T": 12, "people": ex()}
    S["exchange_places_iou_only"] = {"emb": False, "T": 12, "people": ex()}
    S["passer_by_near"] = {"emb": True, "T": 8, "people": [walk_across(220, .3, 1.0, 3.5, name="P", role="passer")]}
    S["passer_by_far"] = {"emb": True, "T": 8, "people": [walk_across(220, .12, 1.0, 3.5, name="P", role="passer")]}
    S["lingering_far"] = {"emb": True, "T": 40, "people": [approach_and_stay(300, .12, 3, 40, name="L", role="far", start_ratio=.1)]}
    S["lingering_near"] = {"emb": True, "T": 20, "people": [approach_and_stay(300, .3, 4, 20, name="V", role="visitor")]}
    S["group_arrives_together"] = {"emb": True, "T": 20, "people": [
        approach_and_stay(200, .3, 5, 20, name="A1", band="adult", t_start=0.0), approach_and_stay(300, .3, 5.4, 20, name="A2", band="adult", t_start=0.4, sign=-1),
        approach_and_stay(400, .3, 5.8, 20, name="C1", band="child", t_start=0.8, scale=cs)]}
    S["adult_at_distance"] = {"emb": True, "T": 20, "people": [approach_and_stay(300, .27, 4, 20, name="V", role="visitor")]}
    S["child_same_distance"] = {"emb": True, "T": 20, "people": [approach_and_stay(300, .27, 4, 20, name="K", role="visitor", band="child", scale=cs)]}
    S["leaves_after_trigger"] = {"emb": True, "T": 40, "people": [Person("V", "visitor", "adult",
        path((0, 300, 220, .08), (4, 300, 220, .3), (8, 300, 220, .3), (8, None, 0, 0), (15, 300, 220, .3), (40, 300, 220, .3)))]}
    sib = lambda use_emb, rel: {"emb": use_emb, "T": 22, "people": [
        Person("S1", "visitor", "adult", path((0, 200, 220, .3), (6, 200, 220, .3), (6, None, 0, 0), (13, 440, 220, .3), (22, 440, 220, .3)), sign=1),
        Person("S2", "visitor", "adult", path((0, 440, 220, .3), (6, 440, 220, .3), (6, None, 0, 0), (13, 200, 220, .3), (22, 200, 220, .3)), sign=-1, relative_of="S1" if rel else None)]}
    S["relatives_return_swapped"] = sib(True, True)
    S["unrelated_return_swapped"] = sib(True, False)
    return S


# ── one run ──────────────────────────────────────────────────────────────────────────────

class _Oracle:
    """Age estimator that reports each person's TRUE band (confident), looked up by box, so group context sees real bands."""
    def estimate(self, frame, box, key=None):
        for p in frame["people"]:
            if p["box"] == box:
                return AgeProbs(0.9, 0.08, 0.02) if p["band"] == "child" else AgeProbs(0.06, 0.92, 0.02)
        return AgeProbs(0.4, 0.4, 0.2)


def _unit(v):
    return v / np.linalg.norm(v)


def run_scenario(name: str, seed: int, c: Optional[dict] = None, log_dir: Optional[str] = None) -> dict:
    c = c or cfg()
    spec = scenarios(c)[name]
    rng = np.random.RandomState(seed)
    people: List[Person] = spec["people"]
    base: Dict[str, np.ndarray] = {}
    for p in people:
        v = _unit(rng.randn(512))
        if p.relative_of:
            o = _unit(rng.randn(512)); r = c["relative_cos"]
            v = _unit(r * base[p.relative_of] + np.sqrt(1 - r * r) * _unit(o - (o @ base[p.relative_of]) * base[p.relative_of]))
        base[p.name] = v
    tan = np.tan(np.arccos(c["genuine_cos"]))
    offs = {p.name: rng.uniform(-c["time_offset_s"], c["time_offset_s"]) for p in people}   # so seeds change the scene, not only the embeddings
    fps, H, W = c["fps"], c["frame_h"], c["frame_w"]
    pipe = AgeBandPipeline(_Oracle(), lambda fr: [Detection(p["box"], p["emb"]) for p in fr["people"]])
    vp = VisitProfile(f"trk-{name}-{seed}")
    grp = GroupTracker()
    eng = EngagementEngine(grp.context, grp.members)
    vp.add_cleanup(pipe.clear); vp.add_cleanup(grp.clear)
    state = {"t": 0.0}
    log = DecisionLog(log_dir, "profile", clock=lambda: state["t"]) if log_dir else None
    m = {k: 0.0 for k in ("false_triggers", "missed_triggers", "id_swaps", "wrong_merges", "fragmented_returns", "writes_total", "writes_dropped",
                          "writes_misattributed", "events", "regreets", "group_one_event", "tte_sum", "tte_n", "visitors", "disengaged", "returned_logged")}
    owner, last_ident, last_seen, ids_of = {}, {}, {}, {}
    ever: Dict[str, set] = {p.name: set() for p in people}
    event_idents: List[set] = []
    event_times: List[float] = []
    first_close: Dict[str, float] = {}
    next_write = 0.0
    maxmiss = pipe.cfg["track_max_missed"]
    F = int(spec["T"] * fps)
    for f in range(F):
        t = f / fps
        state["t"] = t
        frame_people = []
        for p in people:
            pos = p.traj(t - offs[p.name])
            if pos is None:
                continue
            cx, cy, ratio = pos
            cx, cy = cx + rng.randn() * c["pos_jitter_px"], cy + rng.randn() * c["pos_jitter_px"]
            ratio = ratio * (1 + rng.randn() * c["ratio_jitter"])
            h = max(8, int(ratio * H)); w = h
            emb = None
            if spec["emb"]:
                emb = _unit(base[p.name] + tan * _unit(rng.randn(512)))
            frame_people.append({"name": p.name, "band": p.band, "box": (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)), "emb": emb, "ratio": ratio})
            if ratio >= engagement.DEFAULTS["enter_ratio"] and p.name not in first_close:
                first_close[p.name] = t
        res = pipe.process({"people": frame_people})
        for r, fp in zip(res, frame_people):
            tid, ident = r["person_id"], fp["name"]
            if tid not in owner:
                owner[tid] = ident
            if tid in last_seen and f - last_seen[tid] > maxmiss and owner[tid] != ident:
                m["wrong_merges"] += 1
            if tid in last_ident and last_ident[tid] != ident:
                m["id_swaps"] += 1
            last_ident[tid], last_seen[tid] = ident, f
            ids_of.setdefault(ident, set()).add(tid)
        vp.sync_tracks(res, pipe.cfg["stable_frames"])
        for pid, ctx in grp.update(t, res).items():
            if pid in vp.persons:
                vp.set_group_context(pid, ctx)
        obs = [{"person_id": r["person_id"], "box": r["box"], "track_conf": r["track_conf"], "eligible": r["person_id"] in vp.persons} for r in res]
        for ev in eng.update(t, H, obs):
            idents = {owner[i] for i in ev.person_ids}
            event_idents.append(idents); event_times.append(t)
            m["events"] += 1
            if any(next(p for p in people if p.name == i).role in ("passer", "far") for i in idents):
                m["false_triggers"] += 1
            if log:
                for i in ev.person_ids:
                    rr = next((r for r in res if r["person_id"] == i), None)
                    if rr: vp.set_prior(i, rr["band"], rr["confidence"])
                op = opening_for_event(ev, vp)
                log.log_event("engage", persons=list(ev.person_ids), group=ev.group.to_dict(), distance_proxy=ev.distance_proxy,
                              opening_id=op.row_id, prior=vp.prior(ev.person_ids[0]), track_conf=vp.__dict__["_track_conf"].get(ev.person_ids[0], 1.0),
                              suspended_writes=vp.suspended_writes, reason=ev.reason)
        for e in eng.drain_log():
            if e["kind"] == "disengaged": m["disengaged"] += 1
            if e["kind"] == "disengaged_returned": m["returned_logged"] += 1
        if t >= next_write:                                            # one simulated evidence write per tracked visitor per period
            next_write += c["write_every_s"]
            for r, fp in zip(res, frame_people):
                tid = r["person_id"]
                if tid in vp.persons:
                    sign = next(p for p in people if p.name == fp["name"]).sign
                    m["writes_total"] += 1
                    out = vp.record_style(tid, "pace", sign, "llm_statement", f)
                    if out is None:
                        m["writes_dropped"] += 1
                    elif owner[tid] != fp["name"]:
                        m["writes_misattributed"] += 1
    vis = [p for p in people if p.role == "visitor"]
    m["visitors"] = len(vis)
    for p in vis:
        got = [i for i, idents in enumerate(event_idents) if p.name in idents]
        if not got:
            m["missed_triggers"] += 1
        else:
            m["regreets"] += len(got) - 1
            if p.name in first_close:          # seconds from the first frame the TRUE ratio was >= enter_ratio to the first event covering them
                m["tte_sum"] += event_times[got[0]] - first_close[p.name]; m["tte_n"] += 1
    m["fragmented_returns"] = sum(max(0, len(s) - 1) for s in ids_of.values())      # extra ids a single true person received
    if name == "group_arrives_together":
        m["group_one_event"] = float(len(event_idents) == 1 and event_idents[0] == {p.name for p in people})
    if log: log.close()
    vp.end()
    return m


METRICS = ["false_triggers", "missed_triggers", "id_swaps", "wrong_merges", "fragmented_returns", "writes_total", "writes_dropped",
           "writes_misattributed", "events", "regreets", "disengaged", "returned_logged", "group_one_event", "tte_sum", "tte_n"]


def run_all(seeds: List[int], c: dict, log_dir: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for name in scenarios(c):
        arr = {k: np.zeros(len(seeds)) for k in METRICS}
        for i, s in enumerate(seeds):
            r = run_scenario(name, s, c, log_dir if (i == 0) else None)
            for k in METRICS:
                arr[k][i] = r[k]
        out[name] = arr
    return out


def git_info() -> dict:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip(),
                "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=here, text=True).strip())}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


def fmt(t) -> str:
    est, (lo, hi) = t
    return "n/a" if est != est else f"{est:.2f} [{lo:.2f}, {hi:.2f}]"


def render_md(meta: dict, table: Dict[str, Dict[str, tuple]]) -> str:
    cols = ["false_triggers", "missed_triggers", "id_swaps", "wrong_merges", "fragmented_returns", "writes_total", "writes_dropped", "writes_misattributed", "events", "regreets", "group_one_event", "time_to_engage_s"]
    L = [f"# Tracker / trigger simulator results ({meta['seeds'][2]} seeds {meta['seeds'][0]}..{meta['seeds'][1]})", "",
         f"- config hash (sha256 of all effective museum configs): `{meta['config_hash']}`", f"- git: `{meta['git']['commit']}` dirty={meta['git']['dirty']}",
         "- Scripted people, synthetic face embeddings and my own movement model. Numbers are per run (one scene), mean [95% bootstrap CI over seeds].",
         "- Synthetic embeddings show the logic works, not that real faces separate (relatives especially). Nothing here measures real visitors.", "",
         "| scenario | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, d in table.items():
        L.append(f"| {name} | " + " | ".join(fmt(d[k]) for k in cols) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="museum_tracking")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--print-hash", action="store_true")
    a = ap.parse_args(argv)
    if a.print_hash:
        print(config_hash()); return 0
    c = cfg()
    seeds = list(range(a.first_seed, a.first_seed + (a.seeds or c["seeds"])))
    res = run_all(seeds, c, a.log_dir)
    rng = np.random.RandomState(12345)
    table = {}
    for n, arr in res.items():
        table[n] = {k: boot_ci(arr[k], None, c["boot"], rng) for k in METRICS if k not in ("tte_sum", "tte_n")}
        table[n]["time_to_engage_s"] = boot_ci(arr["tte_sum"], arr["tte_n"], c["boot"], rng)
    meta = {"config_hash": config_hash(), "git": git_info(), "seeds": [seeds[0], seeds[-1], len(seeds)]}
    md = render_md(meta, table)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, a.prefix + ".md"), "w").write(md)
        json.dump({"meta": meta, "table": {n: {k: [float(v[0]), [float(v[1][0]), float(v[1][1])]] for k, v in d.items()} for n, d in table.items()}},
                  open(os.path.join(a.out, a.prefix + ".json"), "w"), indent=1)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
