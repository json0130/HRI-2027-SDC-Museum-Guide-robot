import re

import pytest

from .config import load_config
from .initiation import DEFAULTS, choose_opening

AGE_WORDS = re.compile(r"\b(young|old|kid|kids|child|children|adult|little|grown|age|you look|you seem)\b", re.I)


@pytest.mark.parametrize("band,conf", [("unsure", 0.99), ("child", 0.5), ("adult", 0.69), ("child", 0.0)])
def test_unsure_or_low_conf_is_neutral(band, conf):
    assert choose_opening(band, conf, 1).row_id == "neutral_solo"
    assert choose_opening(band, conf, 3).row_id == "neutral_group"


def test_confident_bands_select_their_rows():
    assert choose_opening("child", 0.9, 1).row_id == "child_solo"
    assert choose_opening("adult", 0.9, 4).row_id == "adult_group"
    c = choose_opening("child", 0.9, 1)
    assert (c.vocabulary, c.gaze_height) == ("simple", "low") and c.speech_rate < 1.0


def test_fixed_baseline_ignores_prior():
    assert choose_opening("child", 0.99, 1, use_prior=False) == choose_opening("unsure", 0.0, 1)


def test_neutral_group_inherits_default_fields():
    n = choose_opening("unsure", 0, 2)
    d = DEFAULTS["default"]
    assert n.first_question != "" and n.speech_rate == 1.0 and n.vocabulary == "standard"
    assert "everyone" in n.greeting_template


def test_every_row_is_choice_question_and_never_mentions_age():
    cfg = load_config("museum_initiation", DEFAULTS)
    for key in [k for k in cfg if k not in ("settings", "default")]:
        if key == "group_child_present":                # the group-context wiring row (see tests_group.py)
            from .group import GroupContext
            p = choose_opening("adult", 1.0, 3, cfg=cfg, group=GroupContext("g", 3, True, "known", ""))
        else:
            p = choose_opening(key.split("_")[0], 1.0, 2 if key.endswith("group") else 1, cfg=cfg)
        assert p.row_id == key
        assert " or " in p.first_question and p.first_question.endswith("?"), key
        assert not AGE_WORDS.search(p.greeting_template + " " + p.first_question), key
        assert "{robot_name}" in p.greeting_template and p.greeting("Bot").startswith(("Hello", "Hi", "Good"))


def test_to_dict_json_ready():
    import json
    json.dumps(choose_opening("adult", 0.9, 1).to_dict())
