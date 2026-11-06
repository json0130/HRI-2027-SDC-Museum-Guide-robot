"""Initiation policy: (age band, confidence, group size) -> how the robot OPENS.

Table-driven (config/museum_initiation.yaml, same shape as Library-Book-Robot styles.yaml).
The output is a robot-agnostic dataclass; a Pepper/ChatBox adapter maps it to speech and
motion. The age band only picks a row; it is never put in the wording. `unsure` or low
confidence always selects the neutral row, and `use_prior=False` (baseline) forces it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from .config import load_config

DEFAULTS = {
    "settings": {"min_conf": 0.7},
    "default": {
        "greeting_template": "Hello, and welcome! I'm {robot_name}.",
        "first_question": "Would you like the short version, or the full story?",
        "vocabulary": "standard", "speech_rate": 1.0, "approach_distance_m": 1.5, "gaze_height": "level",
    },
}


@dataclass(frozen=True)
class OpeningParams:
    row_id: str
    greeting_template: str      # contains {robot_name}
    first_question: str         # always a low-cost choice question
    vocabulary: str             # simple | standard
    speech_rate: float
    approach_distance_m: float
    gaze_height: str            # level | low

    def greeting(self, robot_name: str = "your guide") -> str:
        return self.greeting_template.format(robot_name=robot_name)

    def to_dict(self) -> dict:
        return asdict(self)


def choose_opening(band: str, confidence: float, group_size: int, *,
                   use_prior: bool = True, cfg: Optional[dict] = None, group=None) -> OpeningParams:
    """`group` is an optional modules.museum.group.GroupContext (weak, opening-only, never spoken). When given it
    replaces `group_size`: unknown context -> the plain neutral solo opening (no age row either); a known group of 2+
    ignores the individual age band and only uses has_child_band_member (True -> the one placeholder row
    `group_child_present`, else neutral_group); a known solo visitor uses the age-band rows as before."""
    cfg = cfg if cfg is not None else load_config("museum_initiation", DEFAULTS)
    if group is not None:
        if group.status != "known":
            use_prior, group_size = False, 1
        else:
            group_size = group.group_size
            if group_size >= 2:
                if group.has_child_band_member and use_prior:
                    f = {**cfg["default"], **(cfg.get("group_child_present") or {})}
                    return OpeningParams("group_child_present", f["greeting_template"], f["first_question"], f["vocabulary"],
                                         float(f["speech_rate"]), float(f["approach_distance_m"]), f["gaze_height"])
                use_prior = False
    confident = use_prior and band in ("child", "adult") and confidence >= cfg["settings"]["min_conf"]
    row = band if confident else "neutral"
    key = f"{row}_{'group' if group_size >= 2 else 'solo'}"
    f = {**cfg["default"], **(cfg.get(key) or {})}
    return OpeningParams(key, f["greeting_template"], f["first_question"], f["vocabulary"],
                         float(f["speech_rate"]), float(f["approach_distance_m"]), f["gaze_height"])
