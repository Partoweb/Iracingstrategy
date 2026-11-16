"""Stint protocol: says what to drive next and for how many laps.

Default: one push stint, one fuel-save stint at a stated target, and one timed
pit stop. A third stint is asked for only when an open decision needs it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .fitter import PUSH, Fit, prepare_laps
from .solver import Race, Result

STINT_LAPS = 10             # clean laps a protocol stint should have
MAX_REALISTIC_SAVING = 0.15


@dataclass
class Step:
    title: str
    done: bool
    detail: str


@dataclass
class Protocol:
    steps: list[Step] = field(default_factory=list)
    next_text: str = ""

    @property
    def complete(self) -> bool:
        return all(step.done for step in self.steps)


def _best_stint(df: pd.DataFrame, tag: str) -> tuple[int, float | None]:
    """Most clean laps in one stint under this tag, and that stint's start fuel."""
    rows = df[df["usable"] & (df["stint_tag"] == tag)]
    if rows.empty:
        return 0, None
    sizes = rows.groupby(["session_id", "stint_number"]).size()
    session_id, stint = sizes.idxmax()
    stint_rows = df[(df["session_id"] == session_id) & (df["stint_number"] == stint)]
    return int(sizes.max()), float(stint_rows["fuel_start_l"].max())


def save_target(push_burn: float, tank_l: float | None, race: Race | None,
                lap_time: float | None) -> tuple[float, str]:
    """A fuel-per-lap target for the fuel-save stint, and why."""
    fallback = round(push_burn * 0.95, 2)
    if race is None or tank_l is None:
        return fallback, "about 5% under your push average, as a first target"
    if race.timed:
        if not lap_time:
            return fallback, "about 5% under your push average, as a first target"
        laps = math.floor(race.minutes * 60.0 / lap_time) + 1
        margin = race.reserve_laps + race.timed_extra_laps
    else:
        laps, margin = race.laps, race.reserve_laps
    stints_on_push = math.ceil((laps + margin) * push_burn / tank_l)
    if stints_on_push < 2:
        return fallback, ("this race needs no stop on push, so saving cannot drop one; "
                          "5% under push is a generic target")
    target = tank_l * (stints_on_push - 1) / (laps + margin)
    saving = 1.0 - target / push_burn
    if saving > MAX_REALISTIC_SAVING:
        return fallback, (f"dropping a stop in this race would take {saving:.0%} less fuel, "
                          f"which is out of reach; 5% under push is a generic target")
    # Round down so that hitting the target really drops the stop.
    target = math.floor(target * 100) / 100
    stops = stints_on_push - 1
    return target, (f"{saving:.0%} under push: the level that turns {stops} stop"
                    f"{'s' if stops > 1 else ''} into {stops - 1} in this race")


def stint_protocol(laps: pd.DataFrame, fit: Fit, *, tank_l: float | None = None,
                   race: Race | None = None, result: Result | None = None) -> Protocol:
    """Work out which protocol stints are done and what to drive next."""
    df = prepare_laps(laps) if len(laps) else None
    proto = Protocol()
    todo: list[str] = []

    push_laps, push_fuel = _best_stint(df, "push") if df is not None else (0, None)
    push_done = push_laps >= STINT_LAPS
    load = f"{tank_l:g} L (full tank)" if tank_l else "a full tank"
    proto.steps.append(Step("Push stint", push_done,
                            f"{push_laps} clean laps in the longest stint, {STINT_LAPS} wanted"))
    if not push_done:
        todo.append(f"Drive a push stint: {load}, new tires, {STINT_LAPS} clean laps after "
                    f"the out-lap. Tag it first: push")

    save_laps, _ = _best_stint(df, "fuel-save") if df is not None else (0, None)
    save_done = save_laps >= STINT_LAPS
    proto.steps.append(Step("Fuel-save stint", save_done,
                            f"{save_laps} clean laps in the longest stint, {STINT_LAPS} wanted"))
    if not save_done:
        if PUSH in fit.modes:
            lap_time = fit.coef("base") if fit.has_lap_model else None
            target, why = save_target(fit.modes[PUSH].fuel_per_lap, tank_l, race, lap_time)
            same = f"{push_fuel:.0f} L" if push_fuel else load
            todo.append(f"Drive a fuel-save stint: {same} as in the push stint, new tires, "
                        f"{STINT_LAPS} clean laps, holding {target:.2f} L per lap ({why}). "
                        f"Tag it first: fuel-save {target:.2f}")
        else:
            todo.append("Then a fuel-save stint from the same fuel load; the target comes "
                        "from the push stint.")

    stop_done = fit.pit is not None
    proto.steps.append(Step("Timed pit stop", stop_done,
                            f"{fit.pit.n_stops} usable stop(s)" if fit.pit else "none logged"))
    if not stop_done:
        todo.append("Make one pit stop during a push stint: at least 3 clean laps before and "
                    "after, take fuel, and change tires if you will in the race.")

    # The third stint: only when an open decision needs it.
    if fit.fuel_weight_fitted:
        proto.steps.append(Step("Second fuel load", True, "fuel-weight effect measured"))
    elif not (push_done and save_done and stop_done):
        proto.steps.append(Step("Third stint", True, "decided once the first three are done"))
    elif result is None:
        proto.steps.append(Step("Third stint", True,
                                "not needed so far; give the race length to check"))
    elif result.open_decisions or any("second fuel load" in n for n in result.notes):
        reason = result.open_decisions[0] if result.open_decisions else \
            next(n for n in result.notes if "second fuel load" in n)
        proto.steps.append(Step("Second fuel load", False, reason))
        base_load = push_fuel or tank_l or 0.0
        low = max(0.0, base_load - 0.4 * (tank_l or base_load))
        todo.append(f"Drive one more push stint from about {low:.0f} L, new tires, "
                    f"{STINT_LAPS} clean laps. Tag it: push")
    else:
        proto.steps.append(Step("Third stint", True, "not needed: no open decision in this race"))

    proto.next_text = todo[0] if todo else "Nothing more to drive for this race. Run: strategist plan"
    return proto


def render(proto: Protocol) -> str:
    lines = []
    for step in proto.steps:
        lines.append(f"[{'x' if step.done else ' '}] {step.title}: {step.detail}")
    lines.append("")
    lines.append(f"Next: {proto.next_text}")
    return "\n".join(lines)


def stint_progress(laps: pd.DataFrame, session_id: int, stint_number: int) -> str:
    """One line on the stint being driven, for the logger's console."""
    rows = laps[(laps["session_id"] == session_id) & (laps["stint_number"] == stint_number)]
    clean = rows[(rows["clean"] == 1) & (rows["traffic"] == 0)]
    if rows.empty:
        return ""
    tag = rows["stint_tag"].iloc[-1]
    if tag == "untagged":
        return "stint is untagged: its laps will not be used for fitting"
    count = len(clean)
    if count < STINT_LAPS:
        return f"{tag} stint: {count}/{STINT_LAPS} clean laps, {STINT_LAPS - count} to go"
    if count == STINT_LAPS:
        return f"{tag} stint: {count} clean laps, enough. Pit when you like."
    return ""
