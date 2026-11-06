"""Engagement trigger. Synthetic frames: they show the rules behave as designed. The distance proxy is a box ratio, not a
distance, and 0.20 / 0.15 are placeholders."""
import pytest

from .engagement import EngagementEngine, opening_for_event
from .group import GroupContext, GroupTracker, UNKNOWN
from .profile import VisitProfile

H = 480


def box(ratio, x=100):
    h = int(ratio * H)
    return (x, 50, x + h, 50 + h)


def K(gid="group_1", n=1, child=False):
    return GroupContext(gid, n, child, "known", "")


class Rig:
    """Engine + scripted group context. frames: list of {pid: ratio or (ratio, track_conf, eligible)}."""
    def __init__(self, ctx=None, members=None, cfg=None):
        self.ctx = ctx or {}
        self.members = members or {}
        self.eng = EngagementEngine(lambda p: self.ctx.get(p, UNKNOWN), lambda g: self.members.get(g, []), cfg)
        self.t, self.fired = 0.0, []

    def frame(self, obs, dt=0.1):
        o = []
        for pid, v in obs.items():
            r, tc, el = (v, 1.0, True) if not isinstance(v, tuple) else v
            o.append({"person_id": pid, "box": box(r), "track_conf": tc, "eligible": el})
        ev = self.eng.update(self.t, H, o)
        self.fired += ev
        self.t = round(self.t + dt, 6)
        return ev

    def run(self, n, obs, dt=0.1):
        for _ in range(n):
            self.frame(obs, dt)


def test_passer_by_never_triggers():
    r = Rig({"visitor_1": K()})
    r.run(10, {"visitor_1": 0.3}); r.run(30, {})                     # 10 frames close, then gone: never reaches dwell_frames=20
    r.run(15, {"visitor_1": 0.3})
    assert r.fired == []


def test_lingering_far_away_never_triggers():
    r = Rig({"visitor_1": K()})
    r.run(300, {"visitor_1": 0.12})                                    # below exit_ratio the whole time
    r.run(300, {"visitor_1": 0.17})                                    # between exit and enter: holds at zero, never counts
    assert r.fired == []


def test_dwell_fires_once_for_a_known_solo_visitor():
    r = Rig({"visitor_1": K(n=1)}, {"group_1": ["visitor_1"]})
    r.run(19, {"visitor_1": 0.3}); assert r.fired == []
    r.frame({"visitor_1": 0.3})
    assert len(r.fired) == 1
    e = r.fired[0]
    assert (e.person_ids, e.group_id, e.reason, e.distance_proxy) == (("visitor_1",), "group_1", "group_known", 0.3)
    r.run(500, {"visitor_1": 0.3})                                      # staying forever does not re-fire (max_attempts_per_person=1)
    assert len(r.fired) == 1


def test_hysteresis_holds_between_exit_and_enter_and_resets_below_exit():
    r = Rig({"visitor_1": K()}, {"group_1": ["visitor_1"]})
    r.run(10, {"visitor_1": 0.22}); r.run(5, {"visitor_1": 0.17}); r.run(9, {"visitor_1": 0.22})
    assert r.fired == []                                               # 19 counted frames
    r.frame({"visitor_1": 0.22}); assert len(r.fired) == 1             # the 20th, despite the held dip
    r2 = Rig({"visitor_1": K()}, {"group_1": ["visitor_1"]})
    r2.run(10, {"visitor_1": 0.22}); r2.run(1, {"visitor_1": 0.10}); r2.run(15, {"visitor_1": 0.22})
    assert r2.fired == []                                              # a dip below exit_ratio reset the count


def test_low_track_conf_or_no_node_do_not_count():
    r = Rig({"visitor_1": K()}, {"group_1": ["visitor_1"]})
    r.run(100, {"visitor_1": (0.3, 0.3, True)}); assert r.fired == []                  # track_conf below 0.6
    r.run(100, {"visitor_1": (0.3, 1.0, False)}); assert r.fired == []                 # KG node not created yet
    r.run(20, {"visitor_1": (0.3, 1.0, True)}); assert len(r.fired) == 1


def test_group_fires_once_for_everyone_when_the_context_becomes_known():
    r = Rig({}, {"group_1": ["visitor_1", "visitor_2"]})
    r.run(25, {"visitor_1": 0.3, "visitor_2": 0.16})                   # visitor_2 (smaller face) never passes the dwell test itself
    assert r.fired == []                                               # first member passed dwell; context still unknown: waiting
    r.ctx.update(visitor_1=K(n=2, child=True), visitor_2=K(n=2, child=True))
    r.frame({"visitor_1": 0.3, "visitor_2": 0.16})
    assert len(r.fired) == 1 and set(r.fired[0].person_ids) == {"visitor_1", "visitor_2"} and r.fired[0].reason == "group_known"
    assert r.fired[0].group.has_child_band_member is True


def test_settle_timeout_fires_with_unknown_context_and_neutral_opening():
    r = Rig()                                                           # context never becomes known
    r.run(20, {"visitor_1": 0.3, "visitor_2": 0.3})                    # both pass dwell; the first starts the wait
    assert r.fired == []
    while r.t < 20 and not r.fired:
        r.frame({"visitor_1": 0.3, "visitor_2": 0.3})
    e = r.fired[0]
    assert e.reason == "settle_timeout" and e.group.status == "unknown" and e.group_id is None
    assert set(e.person_ids) == {"visitor_1", "visitor_2"} and 5.5 < e.t < 7                  # about group_settle_s (4 s) after dwell
    vp = VisitProfile("v"); vp.add_person("visitor_1"); vp.set_prior("visitor_1", "child", 0.95)
    assert opening_for_event(e, vp).row_id == "neutral_solo"            # unknown context: neutral, even though the prior says child
    vp.end()


def test_pending_is_cancelled_if_the_first_member_leaves_the_zone_before_the_group_settles():
    r = Rig()
    r.run(21, {"visitor_1": 0.3})
    r.run(60, {})                                                       # walked off during the wait
    assert r.fired == []


def test_cooldown_between_events_and_attempts_per_person():
    r = Rig({"visitor_1": K(), "visitor_2": K("group_2")}, {"group_1": ["visitor_1"], "group_2": ["visitor_2"]})
    r.run(20, {"visitor_1": 0.3}); assert len(r.fired) == 1
    r.run(25, {"visitor_2": 0.3})                                       # 2.5 s later, inside cooldown_s=30
    assert len(r.fired) == 1
    while r.t < 40 and len(r.fired) < 2:
        r.frame({"visitor_2": 0.3})
    assert len(r.fired) == 2 and r.fired[1].person_ids == ("visitor_2",) and r.fired[1].t >= 30


def test_leaving_before_the_exchange_disengages_for_good_and_a_return_is_logged_not_greeted():
    r = Rig({"visitor_1": K()}, {"group_1": ["visitor_1"]})
    r.run(20, {"visitor_1": 0.3}); assert len(r.fired) == 1
    r.run(40, {})                                                       # gone 4 s > leave_s (3 s) before any exchange
    assert r.eng.state("visitor_1") == "disengaged"
    r.run(400, {"visitor_1": 0.3})                                      # comes back and lingers, even past the cooldown
    assert len(r.fired) == 1
    kinds = [e["kind"] for e in r.eng.drain_log()]
    assert kinds.count("engage") == 1 and kinds.count("disengaged") == 1 and kinds.count("disengaged_returned") == 1
    assert r.eng.drain_log() == []


def test_leaving_after_the_exchange_started_is_not_a_disengagement():
    r = Rig({"visitor_1": K()}, {"group_1": ["visitor_1"]})
    r.run(20, {"visitor_1": 0.3})
    r.eng.exchange_started(["visitor_1"], r.t)
    r.run(100, {})
    assert r.eng.state("visitor_1") == "exchange"


def test_brief_dropout_does_not_disengage():
    r = Rig({"visitor_1": K()}, {"group_1": ["visitor_1"]})
    r.run(20, {"visitor_1": 0.3}); r.run(10, {}); r.run(5, {"visitor_1": 0.3})        # absent 1 s < leave_s
    assert r.eng.state("visitor_1") == "engaged"


def test_a_childs_smaller_box_fires_later_at_the_same_real_distance():
    """Documented limitation, NOT corrected with the age band: the same approach, faces 0.7x the size, fires later."""
    def approach(scale):
        r = Rig({"visitor_1": K()}, {"group_1": ["visitor_1"]})
        for f in range(200):
            r.frame({"visitor_1": min(0.05 + f * 0.003, 0.5) * scale})        # walking closer: the ratio grows by 0.003 per frame
            if r.fired:
                return f
        return None
    adult, child = approach(1.0), approach(0.7)
    assert adult is not None and (child is None or child > adult + 15)


def test_opening_for_event_uses_prior_and_group_context_at_the_moment():
    vp = VisitProfile("v"); vp.add_person("visitor_1"); vp.add_person("visitor_2")
    vp.set_prior("visitor_1", "adult", 0.95)
    r = Rig({"visitor_1": K(n=1)}, {"group_1": ["visitor_1"]})
    r.run(20, {"visitor_1": 0.3})
    assert opening_for_event(r.fired[0], vp).row_id == "adult_solo"
    r2 = Rig({"visitor_1": K(n=2, child=True), "visitor_2": K(n=2, child=True)}, {"group_1": ["visitor_1", "visitor_2"]})
    r2.run(20, {"visitor_1": 0.3, "visitor_2": 0.3})
    assert opening_for_event(r2.fired[0], vp).row_id == "group_child_present"
    vp.end()


def test_with_the_real_group_tracker_a_pair_arriving_together_gets_one_event_with_a_known_group():
    g = GroupTracker()
    eng = EngagementEngine(g.context, g.members)
    streak, fired, t = {"visitor_1": 0, "visitor_2": 0}, [], 0.0
    while t < 12 and not fired:
        res = []
        for pid, x, t0 in (("visitor_1", 100, 0.0), ("visitor_2", 220, 0.4)):
            if t >= t0:
                streak[pid] += 1
                b = (x, 50, x + 140, 50 + 140)
                res.append({"person_id": pid, "box": b, "band": "adult", "streak": streak[pid], "track_conf": 1.0})
        g.update(t, res)
        obs = [{"person_id": r_["person_id"], "box": r_["box"], "track_conf": 1.0, "eligible": r_["streak"] >= 5} for r_ in res]
        fired += eng.update(t, H, obs)
        t = round(t + 0.1, 6)
    assert len(fired) == 1 and set(fired[0].person_ids) == {"visitor_1", "visitor_2"}
    assert fired[0].group.status == "known" and fired[0].group.group_size == 2 and fired[0].reason == "group_known"


# ── tools/calibrate_engagement.py (pure parts) ───────────────────────────────────────────

def test_calibration_summary_suggestion_and_recording():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("calib", os.path.join(os.path.dirname(__file__), "..", "..", "tools", "calibrate_engagement.py"))
    cal = importlib.util.module_from_spec(spec); spec.loader.exec_module(cal)
    s = cal.summarise([0.30, 0.28, 0.32, 0.31, 0.29, 0.30, 0.33, 0.27, 0.30, 0.31])
    assert s["n"] == 10 and s["min"] == 0.27 and s["max"] == 0.33 and 0.29 <= s["median"] <= 0.31 and s["p10"] <= s["median"] <= s["p90"]
    assert cal.summarise([]) is None
    sug = cal.suggest({1.0: s, 2.0: cal.summarise([0.15, 0.16])}, 1.0)
    assert sug["enter_ratio"] == s["p10"] and sug["exit_ratio"] == round(s["p10"] * 0.75, 3)
    assert cal.suggest({1.0: None}, 1.0) is None
    clock = {"t": 0.0}
    vals = iter([0.3, None, 0.31, 0.29] + [0.3] * 100)
    got = cal.record(lambda: next(vals), 0.5, sleep=lambda dt: clock.__setitem__("t", clock["t"] + dt), now=lambda: clock["t"], period=0.1)
    assert got[:3] == [0.3, 0.31, 0.29] and len(got) == 4                       # None (no face) is skipped
