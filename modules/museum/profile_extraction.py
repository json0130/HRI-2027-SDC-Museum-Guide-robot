"""In-visit, per-turn extraction of likes/dislikes/style cues. App layer: the LLM PROPOSES,
deterministic code below DECIDES what is written (same contract as modules/kg_extraction.py).

Guards: JSON parse (reused ``_parse_json_object``), closed vocab (topic category, style dim,
direction, sign), confidence >= ``CONFIDENCE_MIN``, and the cited ``quote`` must literally occur
in the visitor's turn (anti-hallucination). A parse/shape failure applies nothing; surviving items
are validated first and applied together, never one by one. Inferred mood is never extracted.
The visitor's text is not stored anywhere; only the resulting evidence (no text) is.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from modules.graph_relationship.schema import TOPIC_CATEGORIES
from modules.kg_extraction import CONFIDENCE_MIN, _conf, _parse_json_object

from .profile import DIMS, VisitProfile

LLMFn = Callable[[str, str], str]


def build_system_prompt() -> str:
    dims = "; ".join(f'"{d}" (low={lo}, high={hi})' for d, (lo, hi) in DIMS.items())
    return (
        "You read ONE utterance from a museum visitor and extract only what they explicitly "
        "said about their own preferences. Output ONLY one JSON object, no prose, no code fences:\n"
        '{"topics":[{"label":str,"category":str,"sign":"like"|"dislike","confidence":0-1,"quote":str}],'
        '"style":[{"dim":str,"direction":"low"|"high","confidence":0-1,"quote":str}]}\n'
        f"category must be one of: {', '.join(sorted(TOPIC_CATEGORIES))}.\n"
        f"style dim must be one of: {dims}.\n"
        '"quote" must be copied verbatim from the utterance. Do NOT infer mood, personality, age, '
        "gender, culture or anything not stated. If nothing qualifies, return "
        '{"topics":[],"style":[]}.'
    )


def _clean(label) -> Optional[str]:
    s = re.sub(r"\s+", " ", str(label)).strip()
    return s[:60] if s else None


def parse_update(raw: str, text: str, max_items: int = 3) -> Tuple[Optional[dict], List[tuple]]:
    """-> (update or None, dropped). None means parse/shape failure: apply nothing."""
    obj = _parse_json_object(raw)
    if obj is None or not isinstance(obj.get("topics", []), list) or not isinstance(obj.get("style", []), list):
        return None, [("all", "json_or_shape")]
    low, dropped = text.lower(), []
    topics, style = [], []
    for it in obj.get("topics", []):
        if not isinstance(it, dict): dropped.append(("topic", "not_dict")); continue
        label, cat, sign = _clean(it.get("label", "")), it.get("category"), {"like": 1, "dislike": -1}.get(it.get("sign"))
        if not label or sign is None or cat not in TOPIC_CATEGORIES:
            dropped.append(("topic", "closed_vocab")); continue
        if _conf(it.get("confidence")) < CONFIDENCE_MIN:
            dropped.append(("topic", "low_conf")); continue
        if str(it.get("quote", "")).strip().lower() not in low or not str(it.get("quote", "")).strip():
            dropped.append(("topic", "quote_not_in_text")); continue
        topics.append((label, cat, sign))
    for it in obj.get("style", []):
        if not isinstance(it, dict): dropped.append(("style", "not_dict")); continue
        dim, sign = it.get("dim"), {"high": 1, "low": -1}.get(it.get("direction"))
        if dim not in DIMS or sign is None:
            dropped.append(("style", "closed_vocab")); continue
        if _conf(it.get("confidence")) < CONFIDENCE_MIN:
            dropped.append(("style", "low_conf")); continue
        if str(it.get("quote", "")).strip().lower() not in low or not str(it.get("quote", "")).strip():
            dropped.append(("style", "quote_not_in_text")); continue
        style.append((dim, sign))
    return {"topics": topics[:max_items], "style": style[:max_items]}, dropped


def extract_and_apply_profile(profile: VisitProfile, person_id: str, text: str, llm_fn: LLMFn,
                              turn_ref: Optional[int] = None) -> dict:
    """Run after each turn. Never raises on LLM/parse trouble; returns {applied, topics, style, dropped}."""
    if person_id not in profile.persons or profile.ended:
        return {"applied": False, "topics": 0, "style": 0, "dropped": [("all", "unknown_person_or_ended")]}
    try:
        raw = llm_fn(build_system_prompt(), text)
    except Exception as exc:  # noqa: BLE001 - an LLM outage must not break the visit
        return {"applied": False, "topics": 0, "style": 0, "dropped": [("all", f"llm_error:{type(exc).__name__}")]}
    upd, dropped = parse_update(raw, text, profile.cfg["extraction"]["max_items_per_turn"])
    if upd is None:
        return {"applied": False, "topics": 0, "style": 0, "dropped": dropped}
    for label, cat, sign in upd["topics"]:
        profile.record_topic(person_id, label, cat, sign, "llm_statement", turn_ref)
    for dim, sign in upd["style"]:
        profile.record_style(person_id, dim, sign, "llm_statement", turn_ref)
    return {"applied": bool(upd["topics"] or upd["style"]), "topics": len(upd["topics"]),
            "style": len(upd["style"]), "dropped": dropped}
