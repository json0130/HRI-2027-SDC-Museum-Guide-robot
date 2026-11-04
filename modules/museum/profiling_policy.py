"""Profiling interaction policy: decide the robot's next move toward learning a visitor's
preferences with low regret. Pure and deterministic; the LLM only reaches it through the profile.

Move kinds: none | ask_choice | ask_open | offer | adapt.
  * adapt (confidence >= high): move ONE level on one dimension. No question budget needed.
  * offer (low <= confidence < high): ask first ("Want more detail on this one?"). Counts as a question.
  * below low: keep the generic default; if the budget allows, ask a CHOICE question first,
    and only after every style dimension has had one, a single open interest question.
  * dimension picked = argmax (1 - confidence) x usefulness.
Questions come only from the allow-list in config/museum_policy.yaml (never personal/sensitive).

Small steps: levels live in [-max, +max]. A negative signal on an adapt step (declined, ignored,
contradicted) reverts that step, lowers the dimension's confidence and logs a reversal. Every
decision and outcome is appended to ``step_log`` (JSON-serialisable dicts; persistence is a later
concern). The visual prior never enters here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import load_config
from .profile import DIMS, VisitProfile

DEFAULTS = {
    "thresholds": {"low": 0.35, "high": 0.7, "min_abs_value": 0.3},
    "budget": {"min_turns_between_questions": 3, "max_questions_per_person": 4,
               "ignore_after_turns": 2, "step_cooldown_turns": 2},
    "adaptation": {"max_level": 2, "reversal_conf_factor": 0.5},
    "usefulness": {"talk_level": 1.0, "depth": 0.8, "pace": 0.6, "interest": 0.7},
    "questions": {"choice": {}, "open": "What has caught your eye so far?", "offer": {}},
}
NEGATIVE = ("declined", "ignored", "contradicted")


@dataclass(frozen=True)
class Move:
    kind: str                 # none | ask_choice | ask_open | offer | adapt
    dim: Optional[str] = None
    direction: int = 0        # +1 high end / -1 low end of the dimension
    text: Optional[str] = None
    level: Optional[int] = None   # adapt: the new level


NONE = Move("none")


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class _P:  # per-person policy state
    def __init__(self):
        self.level: Dict[str, int] = {d: 0 for d in DIMS}
        self.asked_choice: set = set()
        self.asked_open = False
        self.questions = 0
        self.last_q_turn = -10**6
        self.last_step_turn: Dict[str, int] = {}
        self.pending: Optional[dict] = None          # the one outstanding question/offer/adapt
        self.adapt: Dict[str, dict] = {}             # dim -> {"level_before","n_evidence"} of last adapt step


class ProfilingPolicy:
    def __init__(self, profile: VisitProfile, cfg: Optional[dict] = None):
        self.profile = profile
        self.cfg = cfg if cfg is not None else load_config("museum_policy", DEFAULTS)
        self._p: Dict[str, _P] = {}
        self.step_log: List[dict] = []

    # -- helpers --------------------------------------------------------------

    def _st(self, pid: str) -> _P:
        return self._p.setdefault(pid, _P())

    def level(self, pid: str, dim: str) -> int:
        return self._st(pid).level[dim]

    def levels(self, pid: str) -> Dict[str, int]:
        return dict(self._st(pid).level)

    def gate(self, confidence: float) -> str:
        t = self.cfg["thresholds"]
        return "generic" if confidence < t["low"] else ("offer" if confidence < t["high"] else "adapt")

    def _log(self, turn, pid, kind, dim, gate, **kw) -> dict:
        row = {"turn": turn, "person": pid, "kind": kind, "dim": dim, "gate": gate, "reversal": False, **kw}
        self.step_log.append(row)
        return row

    def _revert(self, pid, dim, turn, outcome, to_level) -> None:
        st = self._st(pid)
        before = st.level[dim]
        st.level[dim] = to_level
        self.profile.scale_confidence(pid, dim, self.cfg["adaptation"]["reversal_conf_factor"])
        self._log(turn, pid, "revert", dim, "revert", outcome=outcome, level_before=before, level_after=to_level,
                  reversal=True, confidence=self.profile.style(pid, dim)["confidence"])

    # -- feeding outcomes back ------------------------------------------------

    def answer_choice(self, pid: str, dim: str, sign: int, turn: int) -> None:
        """Visitor answered a choice question: explicit, high-weight evidence."""
        st = self._st(pid)
        self.profile.record_style(pid, dim, sign, "choice_answer", turn_ref=turn)
        if st.pending and st.pending["dim"] == dim:
            st.pending = None
        self._log(turn, pid, "answer", dim, self.gate(self.profile.style(pid, dim)["confidence"]),
                  outcome="accepted", direction=sign)

    def observe(self, pid: str, outcome: str, turn: int) -> None:
        """Outcome of the outstanding offer/adapt: accepted | declined | ignored | contradicted.
        (Contradiction of earlier adapt steps is also detected automatically in next_move.)"""
        st = self._st(pid)
        p, st.pending = st.pending, None
        if p is None or p["kind"] not in ("offer", "adapt"):
            return
        dim, d = p["dim"], p["direction"]
        if p["kind"] == "offer":
            if outcome == "accepted":
                self.profile.record_style(pid, dim, d, "choice_answer", turn_ref=turn)
            elif outcome == "declined":
                self.profile.record_style(pid, dim, -d, "choice_answer", turn_ref=turn)
            elif outcome == "ignored":
                self.profile.record_style(pid, dim, -d, "behavior_cue", turn_ref=turn)
            self._log(turn, pid, "offer_outcome", dim, "offer", outcome=outcome, direction=d)
        elif outcome in NEGATIVE:
            if outcome != "contradicted":   # contradictions already arrived as evidence
                self.profile.record_style(pid, dim, -d, "behavior_cue", turn_ref=turn)
            self._revert(pid, dim, turn, outcome, p["level_before"])
            st.adapt.pop(dim, None)
        else:
            self._log(turn, pid, "adapt_outcome", dim, "adapt", outcome=outcome, direction=d)

    # -- deciding -------------------------------------------------------------

    def _expire_and_contradict(self, pid: str, turn: int) -> None:
        st = self._st(pid)
        if st.pending and turn - st.pending["turn"] >= self.cfg["budget"]["ignore_after_turns"]:
            if st.pending["kind"] in ("offer", "adapt"):
                self.observe(pid, "ignored", turn)
            else:
                st.pending = None       # an unanswered plain question: just stop waiting
        for dim, a in list(st.adapt.items()):
            v = self.profile.style(pid, dim)
            if v["evidence_count"] > a["n_evidence"] and st.level[dim] != 0 and _sign(v["value"]) == -_sign(st.level[dim]) \
                    and abs(v["value"]) >= self.cfg["thresholds"]["min_abs_value"]:
                self._revert(pid, dim, turn, "contradicted", st.level[dim] - _sign(st.level[dim]))
                st.adapt.pop(dim, None)

    def next_move(self, pid: str, turn: int) -> Move:
        """Decide (and record) the next move for ``pid`` at ``turn``. Call once per turn."""
        st, c = self._st(pid), self.cfg
        self._expire_and_contradict(pid, turn)
        if st.pending:                       # one outstanding thing at a time
            return NONE
        t, b, mx = c["thresholds"], c["budget"], c["adaptation"]["max_level"]
        info = {d: self.profile.style(pid, d) for d in DIMS}

        # 1. adapt: highest-confidence dimension that can still move one step
        cands = []
        for d, v in info.items():
            dr = _sign(v["value"])
            if dr and abs(v["value"]) >= t["min_abs_value"] and self.gate(v["confidence"]) == "adapt" \
                    and abs(st.level[d] + dr) <= mx and turn - st.last_step_turn.get(d, -10**6) >= b["step_cooldown_turns"]:
                cands.append((v["confidence"], d, dr))
        if cands:
            _, d, dr = max(cands)
            new = st.level[d] + dr
            st.adapt[d] = {"level_before": st.level[d], "n_evidence": info[d]["evidence_count"]}
            st.pending = {"kind": "adapt", "dim": d, "direction": dr, "level_before": st.level[d], "turn": turn}
            st.level[d], st.last_step_turn[d] = new, turn
            self._log(turn, pid, "adapt", d, "adapt", level_before=new - dr, level_after=new, direction=dr,
                      confidence=info[d]["confidence"])
            return Move("adapt", d, dr, None, new)

        # 2. questions (offers count): budget first
        if st.questions >= b["max_questions_per_person"] or turn - st.last_q_turn < b["min_turns_between_questions"]:
            return NONE
        use = c["usefulness"]
        scored = [(  (1 - info[d]["confidence"]) * use[d], d) for d in DIMS]
        if st.asked_choice >= set(DIMS) and not st.asked_open:
            top = max([r["confidence"] for r in self.profile.topics(pid)["likes"]] or [0.0])
            scored.append(((1 - top) * use["interest"], "interest"))
        scored.sort(reverse=True)
        for _, d in scored:
            if d == "interest":
                return self._ask(st, pid, turn, Move("ask_open", None, 0, c["questions"]["open"]), "interest")
            v = info[d]
            g = self.gate(v["confidence"])
            dr = _sign(v["value"])
            if g == "offer" and dr and abs(v["value"]) >= t["min_abs_value"]:
                return self._ask(st, pid, turn, Move("offer", d, dr, c["questions"]["offer"][d]["high" if dr > 0 else "low"]), d)
            if g == "generic" and d not in st.asked_choice:
                st.asked_choice.add(d)
                return self._ask(st, pid, turn, Move("ask_choice", d, 0, c["questions"]["choice"][d]), d)
        return NONE

    def _ask(self, st: _P, pid: str, turn: int, mv: Move, target: str) -> Move:
        if mv.kind == "ask_open":
            st.asked_open = True
        st.questions += 1
        st.last_q_turn = turn
        st.pending = {"kind": mv.kind, "dim": mv.dim, "direction": mv.direction, "turn": turn}
        conf = self.profile.style(pid, mv.dim)["confidence"] if mv.dim else 0.0
        self._log(turn, pid, mv.kind, mv.dim or "interest", self.gate(conf), confidence=conf)
        return mv
