"""Plan view: prints the ranked plans and re-scored adjustments as text."""

from __future__ import annotations

from .fitter import Fit
from .solver import Plan, Race, Result, fuel_text


def clock(seconds: float) -> str:
    """Race time as h:mm:ss.s or m:ss.s."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{sign}{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    return f"{sign}{int(minutes)}:{secs:04.1f}"


def race_line(race: Race, fit: Fit) -> str:
    length = f"{race.minutes:g} min" if race.timed else f"{race.laps} laps"
    rules = []
    if race.mandatory_stops:
        rules.append(f"{race.mandatory_stops} mandatory stop"
                     + ("s" if race.mandatory_stops > 1 else ""))
    if not race.tire_change_allowed:
        rules.append("no tire changes")
    elif race.tire_sets is not None:
        rules.append(f"{race.tire_sets} tire sets")
    rules.append("fuel and tires together" if race.service_together else "tires after fuel")
    return (f"{length} · {fit.car} · {fit.track} · tank {race.tank_l:g} L · "
            + ", ".join(rules))


def plan_lines(plan: Plan, number: int | None = None, *, best: bool = False) -> list[str]:
    head = f"{number}. {plan.title}" if number is not None else plan.title
    if not plan.feasible:
        return [head, *(f"   does not work: {p}" for p in plan.problems)]
    if best:
        tail = plan.verdict or "best"
    else:
        tail = f"{plan.gap_s:+.1f} s ± {plan.gap_u_s:.1f} s"
        if plan.verdict:
            tail += f" · {plan.verdict}"
    lines = [f"{head}    [{tail}]"]
    total = f"   {plan.laps} laps · {clock(plan.time_s)} clear-air"
    if best:
        total += f" ± {plan.gap_u_s:.1f} s"
    lines.append(total)
    lines.append(f"   Start fuel {fuel_text(plan.start_fuel_l)} L")
    for i, stint in enumerate(plan.stints):
        lines.append(f"   Laps {stint.first_lap}-{stint.last_lap}: {stint.mode}")
        if i < len(plan.stops):
            lines.append(f"   {plan.stops[i].line()}")
    for warning in plan.warnings:
        lines.append(f"   note: {warning}")
    return lines


def render(result: Result, race: Race, fit: Fit) -> str:
    """The whole plan view as text."""
    out = [race_line(race, fit)]
    if result.kind == "no-plan":
        out += ["", "No plan.", result.reason]
        out += [f"note: {n}" for n in result.notes]
        return "\n".join(out)

    out.append(f"Based on {result.basis}.")
    out.append("All times are clear-air race time: no traffic, no rivals.")
    out.append("")
    if result.kind == "start-fuel":
        plan = result.plans[0]
        margin = race.reserve_laps + (race.timed_extra_laps if race.timed else 0.0)
        out.append(f"No stop needed. Start fuel: {fuel_text(result.start_fuel_l)} L")
        out.append(f"   covers {plan.laps} laps on push plus {margin:g} laps in hand")
        if race.formation_fuel_l:
            out.append(f"   includes {race.formation_fuel_l:g} L for the laps before the green")
        return "\n".join(out)

    for i, plan in enumerate(result.plans, start=1):
        out += plan_lines(plan, i, best=i == 1)
        out.append("")
    for decision in result.open_decisions:
        out.append(f"open decision: {decision}")
    for note in result.notes:
        out.append(f"note: {note}")
    return "\n".join(out).rstrip()


def render_adjusted(plan: Plan, original_number: int) -> str:
    head = [f"Plan {original_number}, adjusted (gap is against plan 1):"]
    return "\n".join(head + plan_lines(plan))
