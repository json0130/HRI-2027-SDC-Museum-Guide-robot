"""Tracker/trigger simulator tests. Seeds >= 10000 and tiny runs, so the official run (seeds 0..99) is never used for development
or tuning; they assert invariants and scripted expectations, not that the system is good."""
import json
import os

import numpy as np
import pytest

from . import simulate_tracking as st

C = st.cfg()
S = 10000


def run(name, seed=S, **kw):
    return st.run_scenario(name, seed, C, **kw)


def total(name, key, n=12):
    return sum(run(name, S + i)[key] for i in range(n))


def test_deterministic_and_seed_sensitive():
    assert run("relatives_return_swapped") == run("relatives_return_swapped")
    assert any(run("crossing_iou_only", S + i) != run("crossing_iou_only", S) for i in range(1, 6))


def test_embeddings_prevent_the_swaps_that_iou_only_makes():
    assert total("crossing", "id_swaps") == 0
    assert total("crossing_iou_only", "id_swaps") > 0
    assert total("exchange_places", "id_swaps") == 0
    assert total("exchange_places_iou_only", "id_swaps") > 0


def test_silent_swap_is_not_caught_by_write_suspension_and_misattributes_writes():
    r = [run("exchange_places_iou_only", S + i) for i in range(12)]
    assert sum(x["writes_misattributed"] for x in r) > 0          # evidence landed on the wrong person
    assert sum(x["writes_dropped"] for x in r) == 0               # suspension never noticed: track_conf stayed high
    assert total("exchange_places", "writes_misattributed") == 0


def test_detected_crossing_without_embeddings_drops_writes():
    assert total("crossing_iou_only", "writes_dropped") > 0       # overlapping faces w/o embeddings -> ambiguous -> suspended


def test_far_and_lingering_far_never_trigger_and_a_close_visitor_triggers_once():
    assert total("passer_by_far", "events") == 0 and total("lingering_far", "events") == 0
    for i in range(6):
        r = run("lingering_near", S + i)
        assert r["events"] == 1 and r["missed_triggers"] == 0 and r["false_triggers"] == 0


def test_a_passer_by_walking_through_the_near_field_DOES_trigger_documented_weakness():
    """The dwell test counts frames above enter_ratio, not standing still. A person walking across at near range for ~3.5 s
    passes 20 frames: this is a known false trigger, reported as found (not fixed here: the approved spec has no speed gate)."""
    assert total("passer_by_near", "false_triggers") > 0


def test_group_arriving_together_gets_one_event_for_everyone():
    assert all(run("group_arrives_together", S + i)["group_one_event"] == 1 for i in range(8))


def test_child_sized_boxes_miss_at_a_distance_where_an_adult_triggers():
    assert total("adult_at_distance", "missed_triggers") == 0
    assert total("child_same_distance", "missed_triggers") == 12            # documented limitation, not corrected


def test_leaving_after_the_trigger_disengages_and_the_return_is_logged_not_greeted():
    r = [run("leaves_after_trigger", S + i) for i in range(6)]
    assert all(x["events"] == 1 and x["regreets"] == 0 and x["disengaged"] == 1 and x["returned_logged"] == 1 for x in r)


def test_relatives_are_the_riskier_case_for_wrong_merges_than_unrelated_people():
    rel = total("relatives_return_swapped", "wrong_merges", 40) + total("relatives_return_swapped", "fragmented_returns", 40)
    unrel = total("unrelated_return_swapped", "wrong_merges", 40) + total("unrelated_return_swapped", "fragmented_returns", 40)
    assert rel >= unrel                                                       # look-alikes never help; the margin rule fails toward "new id"
    assert total("unrelated_return_swapped", "wrong_merges", 40) == 0


def test_decision_log_for_the_run_is_anonymous_and_the_only_file(tmp_path, monkeypatch):
    work = tmp_path / "w"; work.mkdir(); monkeypatch.chdir(work)
    run("group_arrives_together", log_dir=str(work / "logs"))
    files = [os.path.join(r, f) for r, _, fs in os.walk(work) for f in fs]
    assert len(files) == 1 and files[0].endswith(".jsonl")
    rows = [json.loads(l) for l in open(files[0])]
    e = rows[0]
    assert e["event"] == "engage" and len(e["persons"]) == 3 and e["group"]["status"] == "known" and e["opening_id"] and e["distance_proxy"]
    assert set(e) == {"t", "visit_id", "mode", "event", "persons", "group", "distance_proxy", "opening_id", "prior", "track_conf", "suspended_writes", "reason"}
    assert e["t"] > 0 and len(e["visit_id"]) == 12


def test_config_hash_tracks_effective_config():
    h = st.config_hash()
    o = st.effective_configs(); o["engagement"]["enter_ratio"] = 0.25
    import hashlib
    assert hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest() != h
