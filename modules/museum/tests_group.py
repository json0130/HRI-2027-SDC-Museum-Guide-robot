"""Coarse group context. Synthetic scenes: they show the rules behave as designed, not that real families, couples or
queues are told apart correctly."""
import json

import pytest

from . import profile_api as api
from .group import GroupContext, GroupTracker
from .initiation import choose_opening


class Scene:
    """tracks: {pid: (band, fn(t) -> box | None, first_t)}; drives GroupTracker at 10 Hz, keeping streaks like the tracker."""
    def __init__(self, tracks, cfg=None):
        self.tracks, self.g, self.streak, self.t = tracks, GroupTracker(cfg), {p: 0 for p in tracks}, 0.0

    def run(self, until, dt=0.1):
        out = {}
        while self.t < until:
            res = []
            for pid, (band, fn, t0) in self.tracks.items():
                box = fn(self.t) if self.t >= t0 else None
                if box is None:
                    self.streak[pid] = 0; continue
                self.streak[pid] += 1
                res.append({"person_id": pid, "box": box, "band": band, "streak": self.streak[pid]})
            out = self.g.update(self.t, res)
            self.t = round(self.t + dt, 6)
        return out


def still(x, y=100, w=60):
    return lambda t: (x, y, x + w, y + w)


def walk(x0, vx, y=100, w=60):                 # px per second
    return lambda t: (int(x0 + vx * t), y, int(x0 + vx * t) + w, y + w)


def test_two_people_standing_together_become_a_known_group_after_stable_s():
    s = Scene({"visitor_1": ("adult", still(100), 0.0), "visitor_2": ("adult", still(180), 0.5)})
    early = s.run(2.0)
    assert early["visitor_1"].status == "unknown" and early["visitor_1"].reason in ("settling",)
    late = s.run(8.0)
    c = late["visitor_1"]
    assert (c.status, c.group_size, c.has_child_band_member) == ("known", 2, False)
    assert late["visitor_1"].group_id == late["visitor_2"].group_id


def test_has_child_band_member_true_false_unknown():
    mk = lambda b2: Scene({"visitor_1": ("adult", still(100), 0.0), "visitor_2": (b2, still(180), 0.0)}).run(8.0)["visitor_1"]
    assert mk("child").has_child_band_member is True
    assert mk("adult").has_child_band_member is False
    assert mk("unsure").has_child_band_member is None       # any unsure member: we do not know


def test_far_apart_people_are_separate_groups():
    s = Scene({"visitor_1": ("adult", still(50), 0.0), "visitor_2": ("adult", still(500), 0.0)})
    out = s.run(8.0)
    assert out["visitor_1"].group_size == 1 and out["visitor_2"].group_size == 1 and out["visitor_1"].group_id != out["visitor_2"].group_id


def test_arrival_gap_longer_than_arrival_s_is_not_co_movement():
    s = Scene({"visitor_1": ("adult", still(100), 0.0), "visitor_2": ("adult", still(180), 7.0)})   # arrives 7 s later
    out = s.run(20.0)
    assert out["visitor_1"].group_size == 1 and out["visitor_2"].group_size == 1


def test_passer_by_crossing_a_standing_person_is_not_grouped():
    s = Scene({"visitor_1": ("adult", still(300), 0.0), "visitor_2": ("adult", walk(0, 90), 0.0)})   # one still, one walking past
    out = s.run(8.0)
    assert out["visitor_1"].group_size == 1


def test_two_people_walking_together_are_grouped():
    s = Scene({"visitor_1": ("adult", walk(0, 60), 0.0), "visitor_2": ("adult", walk(80, 60), 0.0)})
    out = s.run(8.0)
    assert out["visitor_1"].group_size == 2 and out["visitor_1"].status == "known"


def test_gap_is_measured_against_the_larger_face_width():
    # adult face 100 px wide next to a child face 40 px wide, 120 px edge-to-edge:
    # 120 / 100 = 1.2 <= 1.5 (together); against the smaller width it would be 3.0 (apart)
    adult = lambda t: (100, 100, 200, 200)
    child = lambda t: (320, 130, 360, 170)
    out = Scene({"visitor_1": ("adult", adult, 0.0), "visitor_2": ("child", child, 0.0)}).run(8.0)
    assert out["visitor_1"].group_size == 2 and out["visitor_1"].has_child_band_member is True


def test_oversized_component_reports_unknown_cap():
    tracks = {f"visitor_{i}": ("adult", still(50 + 70 * i), 0.0) for i in range(1, 9)}   # 8 in a tight row, cap is 6
    out = Scene(tracks).run(10.0)
    c = out["visitor_1"]
    assert (c.status, c.reason, c.group_size, c.has_child_band_member) == ("unknown", "cap", None, None)


def test_merging_two_groups_is_unknown_for_merge_hold_then_known_again():
    # A,B stand left; C,D stand right; after t=12 C,D walk over and stand next to A,B
    def move(x_far, x_near):
        return lambda t: (x_far if t < 12 else x_near, 100, (x_far if t < 12 else x_near) + 60, 160)
    s = Scene({"visitor_1": ("adult", still(100), 0.0), "visitor_2": ("adult", still(170), 0.0),
               "visitor_3": ("adult", move(600, 240), 0.0), "visitor_4": ("adult", move(670, 310), 0.0)})
    mid = s.run(11.5)
    assert mid["visitor_1"].group_size == 2 and mid["visitor_3"].group_size == 2 and mid["visitor_1"].group_id != mid["visitor_3"].group_id
    seen = set()
    while s.t < 40:
        out = s.run(s.t + 0.5)
        if "visitor_1" in out and out["visitor_1"].group_id == out["visitor_3"].group_id:
            seen.add((out["visitor_1"].status, out["visitor_1"].reason))
    assert ("unknown", "merge") in seen                         # reported unknown while merged-and-held
    assert out["visitor_1"].status == "known" and out["visitor_1"].group_size == 4        # and known again afterwards


def test_leaving_member_shrinks_the_group_and_context_is_unknown_while_settling():
    s = Scene({"visitor_1": ("adult", still(100), 0.0), "visitor_2": ("adult", still(180), 0.0),
               "visitor_3": ("adult", lambda t: (260, 100, 320, 160) if t < 10 else None, 0.0)})
    assert s.run(9.0)["visitor_1"].group_size == 3
    seen, out = set(), {}
    while s.t < 30:
        out = s.run(s.t + 0.5)
        seen.add((out["visitor_1"].status, out["visitor_1"].group_size))
    assert ("unknown", None) in seen and out["visitor_1"].group_size == 2 and out["visitor_1"].status == "known"


def test_lone_visitor_is_unknown_until_stable_s_after_arrival():
    s = Scene({"visitor_1": ("adult", still(100), 5.0)})
    first = None
    while s.t < 12:
        out = s.run(s.t + 0.1)
        if "visitor_1" in out and first is None:
            first = out["visitor_1"]
    assert first.status == "unknown" and out["visitor_1"].status == "known" and out["visitor_1"].group_size == 1


def test_unstable_tracks_are_not_members():
    s = Scene({"visitor_1": ("adult", still(100), 0.0)}, cfg={"stable_frames": 50})
    assert s.run(2.0) == {}                                      # fewer than 50 consecutive frames: nobody is a member yet


def test_clear_drops_everything():
    s = Scene({"visitor_1": ("adult", still(100), 0.0)}); s.run(5.0)
    s.g.clear()
    assert s.g.context("visitor_1").reason == "no_track" and s.g.members("group_1") == []


# ── wiring: opening, profile, API ────────────────────────────────────────────────────────

K = lambda n, child: GroupContext("group_1", n, child, "known", "")
U = GroupContext("group_1", None, None, "unknown", "settling")


def test_choose_opening_with_group_context():
    assert choose_opening("child", 0.95, 1, group=U).row_id == "neutral_solo"            # unknown: plain neutral, age row ignored too
    assert choose_opening("adult", 0.95, 1, group=K(1, False)).row_id == "adult_solo"    # known solo: band rows as before
    assert choose_opening("adult", 0.95, 3, group=K(3, False)).row_id == "neutral_group"  # known group: band ignored
    assert choose_opening("adult", 0.95, 3, group=K(3, None)).row_id == "neutral_group"
    o = choose_opening("adult", 0.95, 3, group=K(3, True))
    assert o.row_id == "group_child_present" and "{robot_name}" in o.greeting_template
    assert choose_opening("child", 0.95, 1, use_prior=False, group=K(3, True)).row_id == "neutral_group"   # fixed baseline ignores it
    assert choose_opening("child", 0.95, 1).row_id == "child_solo"                         # no group arg: unchanged behaviour


def test_group_wording_never_mentions_relationships_or_age():
    o = choose_opening("adult", 0.95, 3, group=K(3, True))
    text = (o.greeting_template + " " + o.first_question).lower()
    assert not any(w in text for w in ("family", "kids", "children", "child", "parent", "couple", "friends", "young", "age"))


def test_group_context_hidden_by_default_opening_only_dropped_after_evidence_and_not_in_kg():
    vp = api.start_visit("g1")
    vp.add_person("visitor_1"); vp.add_person("visitor_2")
    vp.set_group_context("visitor_1", K(2, True)); vp.set_group_context("visitor_2", K(2, True))
    vp.record_style("visitor_2", "pace", 1, "choice_answer")
    default = api.get_visit_snapshot("g1")
    assert all("group" not in p for p in default["persons"].values()) and "group_1" not in json.dumps(default)
    inc = api.get_visit_snapshot("g1", include_group=True)["persons"]
    assert inc["visitor_1"]["group"] == {"group_id": "group_1", "group_size": 2, "has_child_band_member": True,
                                         "status": "known", "reason": "", "use": "opening_only"}
    assert inc["visitor_2"]["group"] is None                                              # real evidence exists: dropped
    assert "group_1" not in json.dumps([n.model_dump(mode="json") for n in vp.store._nodes.values()] +
                                       [e.model_dump(mode="json") for e in vp.store._edges.values()])   # never a KG fact
    api.end_visit("g1")


def test_a_group_arriving_together_is_never_reported_as_a_known_solo_first():
    s = Scene({"visitor_1": ("adult", still(100), 0.0), "visitor_2": ("child", still(180), 0.4), "visitor_3": ("adult", still(260), 0.8)})
    solo_known, final = False, {}
    while s.t < 12:
        final = s.run(s.t + 0.1)
        solo_known |= any(c.status == "known" and c.group_size == 1 for c in final.values())
    assert not solo_known and final["visitor_1"].group_size == 3 and final["visitor_1"].has_child_band_member is True
