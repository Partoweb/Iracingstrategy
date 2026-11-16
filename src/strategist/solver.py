"""Solver: race, series rules, fitted parameters and current state in, ranked plans out.

A pure function. It reads no files and makes no network calls, so the live
re-plan can call it with the current lap, fuel and tire age.

The search is brute force over stop count, pit laps, tire changes and pace
mode per stint. Times are clear-air race time: the solver does not see traffic
or rivals.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np

from .fitter import DEFAULT_GROUP, PUSH, Fit

TOO_CLOSE_SIGMAS = 1.0      # a gap inside this many U is too close to call
MAX_PLANS = 5
EVAL_BUDGET = 5_000_000     # pit-lap rows x plan shapes per stop count, before coarsening
PLAUSIBLE_FUEL_WEIGHT = 0.06    # s per litre; starting value for the open-decision check
AGE_SLACK = 1.25            # a stint may run this far past the measured tire age unflagged


@dataclass(frozen=True)
class Race:
    tank_l: float
    laps: int | None = None             # lap-limited race
    minutes: float | None = None        # timed race
    mandatory_stops: int = 0
    tire_sets: int | None = None        # sets available including the starting set
    tire_change_allowed: bool = True
    service_together: bool = True       # fuel and tires go on at the same time
    reserve_laps: float = 0.5           # fuel kept in hand at every stop and at the flag
    timed_extra_laps: float = 1.0       # timed races: fuel for this many laps past the prediction
    white_flag_lap: bool = False        # timed races: one more lap after the lap that ends the clock
    formation_fuel_l: float = 0.0       # burned before the green flag
    max_stops: int = 4

    def __post_init__(self) -> None:
        if (self.laps is None) == (self.minutes is None):
            raise ValueError("give the race length as laps or as minutes, not both")
        if self.tank_l <= 0:
            raise ValueError("tank size must be positive")

    @property
    def timed(self) -> bool:
        return self.minutes is not None


@dataclass(frozen=True)
class State:
    """Where the race stands. The default is before the start."""
    laps_done: int = 0
    fuel_l: float | None = None         # None: fuel is still free to choose
    tire_age: int = 0                   # laps on the current set
    elapsed_s: float = 0.0
    stops_done: int = 0
    tire_sets_used: int = 1


@dataclass(frozen=True)
class PlanSpec:
    """A plan reduced to its decisions. Stint i ends with stop i."""
    lens: tuple[int, ...]               # laps per stint
    modes: tuple[str, ...]              # pace mode per stint
    tires: tuple[bool, ...]             # tire change per stop
    fuel_adds: tuple[float | None, ...] = ()    # litres per stop; None = just enough
    start_fuel: float | None = None     # None = just enough

    @property
    def stops(self) -> int:
        return len(self.lens) - 1

    @property
    def laps(self) -> int:
        return sum(self.lens)

    def pit_laps(self) -> list[int]:
        return list(itertools.accumulate(self.lens))[:-1]


@dataclass
class StintOut:
    first_lap: int
    last_lap: int
    mode: str
    fuel_start_l: float


@dataclass
class StopOut:
    lap: int
    add_l: float
    tires: bool
    stationary_s: float

    def line(self) -> str:
        return (f"Pit lap {self.lap}, add {fuel_text(self.add_l)} L, "
                f"{'tires' if self.tires else 'no tires'}")


@dataclass
class Plan:
    spec: PlanSpec
    title: str
    stints: list[StintOut]
    stops: list[StopOut]
    start_fuel_l: float
    laps: int                           # race laps this plan runs (from the start)
    time_s: float                       # clear-air time from now to the flag
    gap_s: float = 0.0
    gap_u_s: float = 0.0                # uncertainty of the gap (of the time, for the best plan)
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)   # non-empty: the plan does not work
    flip_weight: float | None = None    # fuel weight (s/L) at which this plan would win
    vec: dict[str, float] = field(default_factory=dict, repr=False)
    exposure: float = 0.0
    metric_s: float = 0.0               # time used for ranking (see solve)

    @property
    def feasible(self) -> bool:
        return not self.problems


@dataclass
class Result:
    kind: str                           # 'plans', 'start-fuel' or 'no-plan'
    plans: list[Plan] = field(default_factory=list)
    start_fuel_l: float | None = None
    race_laps: int | None = None
    basis: str = ""
    notes: list[str] = field(default_factory=list)
    open_decisions: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def best(self) -> Plan | None:
        return self.plans[0] if self.plans else None


def fuel_text(litres: float) -> str:
    """Fuel amounts are rounded up so a printed number never under-fuels."""
    return str(int(math.ceil(litres - 1e-6)))


# -- the model the solver computes with -------------------------------------------

class _Model:
    def __init__(self, fit: Fit, race: Race, state: State):
        self.fit = fit
        self.race = race
        self.state = state
        self.modes = sorted(fit.modes, key=lambda m: (m != PUSH, m))
        self.base = fit.coef("base")
        self.cost = {m: (0.0 if m == PUSH else fit.coef(f"cost:{m}")) for m in self.modes}
        self.group = {m: fit.modes[m].slope_group for m in self.modes}
        self.groups = sorted(set(self.group.values()) | {DEFAULT_GROUP})
        self.slope = {g: fit.coef(f"wear:{g}") for g in self.groups}
        self.fuel = {m: fit.modes[m].fuel_per_lap for m in self.modes}
        self.weight = fit.coef("fuel") if fit.fuel_weight_fitted else 0.0
        self.ref = fit.ref_fuel or 0.0
        # One fuel load only: the fit gives the net slope of a stint, tire wear
        # minus fuel burn-off. When that slope is negative, some fuel-weight
        # effect must exist. Use the smallest one that explains it (all of the
        # slope is fuel, none is tire wear); the solver reports separately how
        # much more fuel weight it would take to change the ranking.
        self.fuel_led = (not fit.fuel_weight_fitted) and self.slope[DEFAULT_GROUP] < 0
        if self.fuel_led:
            self.weight = -self.slope[DEFAULT_GROUP] / self.fuel[PUSH]
            self.slope = {g: self.slope[g] + self.weight * self._group_burn(g)
                          for g in self.groups}
        pit = fit.pit
        self.pit_fixed = pit.fixed_loss if pit else None
        self.rate = pit.refuel_rate if pit else None
        self.overhead = pit.overhead if pit else 0.0
        self.tire_time = pit.tire_time if pit else None
        self.together = race.service_together
        self.reserve = race.reserve_laps
        self.extra = race.timed_extra_laps if race.timed else 0.0
        self.live = state.fuel_l is not None
        self.formation = 0.0 if state.laps_done > 0 or self.live else race.formation_fuel_l
        self.cov = np.array(fit.cov, dtype=float)
        self.columns = list(fit.columns)

    def _group_burn(self, group: str) -> float:
        return self.fuel[PUSH] if group == DEFAULT_GROUP else self.fuel[group]

    def to_theta(self, vec: dict[str, float]) -> dict[str, float]:
        """Express a plan's time sensitivities in terms of the fitted parameters."""
        if not self.fuel_led:
            return vec
        out = dict(vec)
        through_weight = out.pop("fuel", 0.0) + sum(
            self._group_burn(g) * vec.get(f"wear:{g}", 0.0) for g in self.groups)
        key = f"wear:{DEFAULT_GROUP}"
        out[key] = out.get(key, 0.0) - through_weight / self.fuel[PUSH]
        return out

    # Fuel a stint needs on board at its first lap.
    def need(self, mode: str, laps, last: bool):
        return self.fuel[mode] * (laps + self.reserve + (self.extra if last else 0.0))

    def stationary(self, add, tires: bool):
        """Standstill for a stop. Works on floats and on arrays."""
        fuel_part = np.where(np.asarray(add) > 1e-9,
                             self.overhead + np.asarray(add) / (self.rate or 1.0), 0.0)
        tire_part = (self.tire_time or 0.0) if tires else 0.0
        if self.together:
            return np.maximum(fuel_part, tire_part)
        return fuel_part + tire_part

    def stint_cap(self, fuel_per_lap: float, *, first: bool, last: bool) -> int:
        """Most laps one stint can run on the fuel it can carry."""
        tank = self.race.tank_l
        if first:
            tank = self.state.fuel_l if self.live else tank - self.formation
        laps = tank / fuel_per_lap - self.reserve - (self.extra if last else 0.0)
        return int(math.floor(laps + 1e-9))

    def min_stops(self, n_laps: int, fuel_per_lap: float, limit: int) -> int | None:
        for stops in range(0, limit + 1):
            if stops == 0:
                reach = self.stint_cap(fuel_per_lap, first=True, last=True)
            else:
                reach = (self.stint_cap(fuel_per_lap, first=True, last=False)
                         + (stops - 1) * self.stint_cap(fuel_per_lap, first=False, last=False)
                         + self.stint_cap(fuel_per_lap, first=False, last=True))
                if min(self.stint_cap(fuel_per_lap, first=True, last=False),
                       self.stint_cap(fuel_per_lap, first=False, last=True)) < 1:
                    continue
            if reach >= n_laps:
                return stops
        return None


# -- vectorised evaluation of one plan shape over many pit-lap choices ------------

def _shape_times(model: _Model, lens: np.ndarray, modes: tuple[str, ...],
                 tires: tuple[bool, ...]):
    """Race time for every row of stint lengths under one shape.

    Returns (total time, feasible mask, time of the last lap, time of the last two laps).
    """
    race, state = model.race, model.state
    rows = lens.shape[0]
    total = np.zeros(rows)
    ok = np.ones(rows, dtype=bool)
    counts = {g: np.zeros(rows) for g in model.groups}
    counts[DEFAULT_GROUP] += state.tire_age
    fuel_prev = n_prev = None
    tail1 = tail2 = None
    last_index = lens.shape[1] - 1
    for k in range(lens.shape[1]):
        n = lens[:, k].astype(float)
        mode = modes[k]
        burn = model.fuel[mode]
        need = model.need(mode, n, last=k == last_index)
        if k == 0:
            if model.live:
                fuel = np.full(rows, float(state.fuel_l))
                ok &= fuel >= need - 1e-9
            else:
                fuel = need
                ok &= need + model.formation <= race.tank_l + 1e-9
        else:
            left = fuel_prev - model.fuel[modes[k - 1]] * n_prev
            add = np.maximum(0.0, need - left)
            fuel = left + add
            ok &= fuel <= race.tank_l + 1e-9
            total += model.pit_fixed + model.stationary(add, tires[k - 1])
            if tires[k - 1]:
                for g in counts:
                    counts[g] = np.zeros(rows)
            else:
                counts[model.group[modes[k - 1]]] = counts[model.group[modes[k - 1]]] + n_prev
        wear = sum(model.slope[g] * counts[g] for g in model.groups)
        level = model.base + model.cost[mode] + wear + model.weight * (fuel - model.ref)
        trend = model.slope[model.group[mode]] - model.weight * burn
        total += n * level + trend * n * (n - 1) / 2
        if k == last_index:
            tail1 = level + trend * (n - 1)
            tail2 = 2 * level + trend * (2 * n - 3)
        fuel_prev, n_prev = fuel, n
    return total, ok, tail1, tail2


def _count_rows(n_laps: int, stops: int, caps: list[int]) -> int:
    """How many ways the pit laps can be placed, without listing them."""
    ways = np.zeros(n_laps + 1)
    ways[0] = 1.0
    laps = np.arange(n_laps + 1)
    for i in range(stops):
        csum = np.concatenate([[0.0], np.cumsum(ways)])
        new = csum[laps] - csum[np.maximum(0, laps - caps[i])]
        rest_cap = sum(caps[i + 1:])
        new[(laps < 1) | (n_laps - laps > rest_cap) | (n_laps - laps < stops - i)] = 0.0
        ways = new
    if stops == 0:
        return int(n_laps <= caps[0])
    return int(ways.sum())


def _pit_lap_rows(n_laps: int, stops: int, caps: list[int], step: int = 1,
                  windows: list[tuple[int, int]] | None = None) -> np.ndarray:
    """All pit-lap choices as an integer array (rows, stops).

    caps holds the longest allowed stint per stint. `step` thins the grid,
    `windows` limits each stop to a range of laps (a locked lap is a window
    of one).
    """
    rows = np.zeros((1, 0), dtype=np.int64)
    prev = np.zeros(1, dtype=np.int64)
    for i in range(stops):
        lo, hi = windows[i] if windows else (1, n_laps - 1)
        lo, hi = max(1, lo), min(n_laps - 1, hi)
        if windows:
            values = np.arange(lo, hi + 1)
        else:
            values = np.arange(((lo + step - 1) // step) * step, hi + 1, step)
        if not len(values):
            return np.zeros((0, stops), dtype=np.int64)
        v, p = values[None, :], prev[:, None]
        rest_cap = sum(caps[i + 1:])
        good = ((v > p) & (v - p <= caps[i]) & (n_laps - v <= rest_cap)
                & (n_laps - v >= stops - i))
        r, c = np.nonzero(good)
        rows = np.column_stack([rows[r], values[c]])
        prev = values[c]
    if stops == 0 and n_laps > caps[0]:
        return np.zeros((0, 0), dtype=np.int64)
    return rows


def _lens_from_pit_laps(pit_laps: np.ndarray, n_laps: int) -> np.ndarray:
    edges = np.column_stack([np.zeros(len(pit_laps), dtype=np.int64), pit_laps,
                             np.full(len(pit_laps), n_laps, dtype=np.int64)])
    return np.diff(edges, axis=1)


# -- exact evaluation of one plan, lap by lap --------------------------------------

def _evaluate(model: _Model, spec: PlanSpec, upto: int | None = None) -> Plan:
    """Walk one plan lap by lap. `upto` stops the clock after that many laps."""
    race, state = model.race, model.state
    stops_n = spec.stops
    adds_in = spec.fuel_adds if spec.fuel_adds else (None,) * stops_n
    problems: list[str] = []
    warnings: list[str] = []
    vec: dict[str, float] = {}
    counts = {g: 0.0 for g in model.groups}
    counts[DEFAULT_GROUP] = float(state.tire_age)
    time = 0.0
    exposure = 0.0
    lap_no = state.laps_done        # race lap number of the last lap driven
    done = 0                        # laps of this plan driven so far
    stints: list[StintOut] = []
    stops: list[StopOut] = []
    fuel = 0.0
    oldest = (0.0, "")

    def bump(key: str, amount: float) -> None:
        vec[key] = vec.get(key, 0.0) + amount

    for k, (n, mode) in enumerate(zip(spec.lens, spec.modes)):
        last = k == stops_n
        if mode not in model.fuel:
            problems.append(f"pace mode {mode!r} has not been measured")
            break
        burn = model.fuel[mode]
        need = model.need(mode, n, last)
        if k == 0:
            if model.live:
                fuel = float(state.fuel_l)
            elif spec.start_fuel is not None:
                fuel = spec.start_fuel - model.formation
            else:
                fuel = need
            if not model.live and fuel + model.formation > race.tank_l + 1e-6:
                problems.append(f"start fuel {fuel + model.formation:.1f} L does not fit "
                                f"the {race.tank_l:g} L tank")
            start_fuel = fuel + model.formation
        else:
            left = fuel
            add = adds_in[k - 1]
            if add is None:
                add = max(0.0, need - left)
            if left + add > race.tank_l + 1e-6:
                problems.append(f"stop {k}: {left + add:.1f} L does not fit the "
                                f"{race.tank_l:g} L tank")
            fuel = left + add
            tires = spec.tires[k - 1]
            if tires and model.tire_time is None:
                problems.append("no tire-change time has been measured")
            if add > 1e-9 and model.rate is None:
                problems.append("no refuel rate has been measured")
            stationary = float(model.stationary(add, tires))
            stops.append(StopOut(lap=lap_no, add_l=add, tires=tires, stationary_s=stationary))
            if upto is None or done < upto:
                time += (model.pit_fixed or 0.0) + stationary
                bump("pit_fixed", 1.0)
                fuel_part = model.overhead + add / model.rate if add > 1e-9 and model.rate else 0.0
                tire_part = (model.tire_time or 0.0) if tires else 0.0
                if add > 1e-9 and model.rate and (not model.together or fuel_part >= tire_part):
                    bump("refuel_rate", -add / model.rate ** 2)
                if tires and (not model.together or tire_part > fuel_part):
                    bump("tire_time", 1.0)
            if tires:
                counts = {g: 0.0 for g in model.groups}

        if fuel < need - 1e-6:
            laps_possible = int(math.floor(fuel / burn + 1e-9))
            if laps_possible < n:
                problems.append(f"runs out of fuel on lap {lap_no + laps_possible + 1}")
            else:
                spare = (fuel - burn * n) / burn
                warnings.append(f"stint {k + 1} ends with {spare:.1f} laps of fuel in hand, "
                                f"less than the planned margin")
        stints.append(StintOut(first_lap=lap_no + 1, last_lap=lap_no + n, mode=mode,
                               fuel_start_l=fuel))
        group = model.group[mode]
        for _ in range(n):
            if upto is None or done < upto:
                wear = sum(model.slope[g] * counts[g] for g in model.groups)
                time += (model.base + model.cost[mode] + wear
                         + model.weight * (fuel - model.ref))
                bump("base", 1.0)
                if mode != PUSH:
                    bump(f"cost:{mode}", 1.0)
                for g in model.groups:
                    bump(f"wear:{g}", counts[g])
                if model.weight:
                    bump("fuel", fuel - model.ref)
                exposure += model.ref - fuel - model.fuel[PUSH] * sum(counts.values())
            counts[group] += 1.0
            fuel -= burn
            lap_no += 1
            done += 1
        age = sum(counts.values())
        measured = model.fit.modes[mode].max_age
        if age > AGE_SLACK * (measured + 1) and age - measured > oldest[0]:
            oldest = (age - measured,
                      f"stint {k + 1} takes a tire set to {age:.0f} laps; {mode} was "
                      f"measured to {measured + 1}")

    if oldest[1]:
        warnings.append(oldest[1])
    if not model.fit.fuel_weight_fitted and not all(spec.tires):
        warnings.append("wear on the used tire set is extrapolated: one fuel load cannot "
                        "separate tire wear from fuel burn-off")
    if any(s.tires for s in stops) and model.fit.pit is not None \
            and model.fit.pit.tire_time_kind == "upper-bound" and vec.get("tire_time"):
        warnings.append("tire-change time is an upper bound, not a measurement")
    title = _title(spec)
    return Plan(spec=spec, title=title, stints=stints, stops=stops,
                start_fuel_l=start_fuel if stints else 0.0,
                laps=state.laps_done + spec.laps, time_s=time, warnings=warnings,
                problems=problems, vec=vec, exposure=exposure, metric_s=time)


def _title(spec: PlanSpec) -> str:
    stops = spec.stops
    head = "No stop" if stops == 0 else f"{stops} stop" + ("s" if stops > 1 else "")
    modes = spec.modes[0] if len(set(spec.modes)) == 1 else " / ".join(spec.modes)
    changes = sum(spec.tires)
    tires = ""
    if stops and changes == 0:
        tires = ", no tire change"
    elif stops and changes < stops:
        tires = f", tires at {changes} of {stops} stops"
    return f"{head}, {modes}{tires}"


def _gap_uncertainty(model: _Model, a: dict[str, float], b: dict[str, float] | None) -> float:
    """Standard error of time(a) - time(b) from the spread of the fitted parameters."""
    keys = set(a) | set(b or {})
    diff = model.to_theta({k: a.get(k, 0.0) - (b.get(k, 0.0) if b else 0.0) for k in keys})
    lap = np.array([diff.get(c, 0.0) for c in model.columns])
    var = float(lap @ model.cov @ lap) if len(lap) else 0.0
    pit = model.fit.pit
    if pit is not None:
        var += (diff.get("pit_fixed", 0.0) * pit.fixed_loss_se) ** 2
        var += (diff.get("refuel_rate", 0.0) * (pit.refuel_rate_se or 0.0)) ** 2
        var += (diff.get("tire_time", 0.0) * (pit.tire_time_se or 0.0)) ** 2
    return math.sqrt(max(0.0, var))


# -- timed races ---------------------------------------------------------------------

def _timed_ok(model: _Model, seconds_left: float, total, tail1, tail2, last_len):
    """Does a plan of this length really end on its last lap? (array or scalar)

    A final stint too short to hold the lap that ends the clock is left out:
    the clock would then run out during the stop itself.
    """
    if model.race.white_flag_lap:
        # The clock runs out on the lap before the last one.
        return ((total - tail2 < seconds_left) & (seconds_left <= total - tail1)
                & (last_len >= 3))
    return (total - tail1 < seconds_left) & (seconds_left <= total) & (last_len >= 2)


def _timed_lap_counts(model: _Model, stops: int, patterns: list[tuple[bool, ...]],
                      seconds_left: float) -> list[int]:
    """Lap counts worth searching for a timed race with this many stops."""
    found: set[int] = set()
    guess = max(stops + 1, int(seconds_left / max(1.0, model.base)))
    for mode in model.modes:
        for tires in patterns[:1] + patterns[-1:]:
            n = guess
            for _ in range(60):
                if n < stops + 1:
                    break
                share, extra = divmod(n, stops + 1)
                lens = np.array([[share + (1 if i < extra else 0) for i in range(stops + 1)]])
                total, _, tail1, tail2 = _shape_times(model, lens, (mode,) * (stops + 1), tires)
                if bool(_timed_ok(model, seconds_left, total, tail1, tail2, lens[:, -1])[0]):
                    found.add(n)
                    break
                reached = total[0] - (tail1[0] if model.race.white_flag_lap else 0.0)
                n += 1 if reached < seconds_left else -1
    out: set[int] = set()
    for n in found:
        out.update((n - 1, n, n + 1))
    return sorted(x for x in out if x >= stops + 1)


# -- the search ----------------------------------------------------------------------

def _tire_patterns(model: _Model, stops: int, notes: list[str]) -> list[tuple[bool, ...]]:
    """Which stops may change tires, under the rules and under what has been measured."""
    race, state = model.race, model.state
    if stops == 0:
        return [()]
    allowed = stops
    if not race.tire_change_allowed:
        allowed = 0
    elif race.tire_sets is not None:
        allowed = max(0, min(stops, race.tire_sets - state.tire_sets_used))
    patterns = [p for p in itertools.product((True, False), repeat=stops) if sum(p) <= allowed]
    if model.tire_time is None:
        patterns = [p for p in patterns if not any(p)]
        if allowed > 0:
            _note(notes, "No stop with a tire change has been timed, so only plans that keep "
                         "the tires were evaluated.")
    return patterns


def _note(notes: list[str], text: str) -> None:
    if text not in notes:
        notes.append(text)


def _search(model: _Model, n_laps: int, stops: int, patterns: list[tuple[bool, ...]],
            seconds_left: float | None, notes: list[str]) -> dict[tuple, tuple[float, PlanSpec]]:
    """Best plan per family (stops, tire changes, laps) for one race length and stop count."""
    best: dict[tuple, tuple[float, PlanSpec]] = {}
    if not patterns:
        return best
    low_burn = min(model.fuel.values())
    caps = [model.stint_cap(low_burn, first=i == 0, last=i == stops) for i in range(stops + 1)]
    if min(caps) < 1:
        return best
    n_rows = _count_rows(n_laps, stops, caps)
    if n_rows == 0:
        return best
    mode_sets = list(itertools.product(model.modes, repeat=stops + 1))
    n_shapes = len(mode_sets) * len(patterns)
    step = 1
    if stops and n_rows * n_shapes > EVAL_BUDGET:
        step = int(math.ceil((n_rows * n_shapes / EVAL_BUDGET) ** (1.0 / stops)))
        _note(notes, f"The {stops}-stop search was run on a {step}-lap grid first and then "
                     f"refined, to keep it quick.")
    pit_laps = _pit_lap_rows(n_laps, stops, caps, step=step)
    if not len(pit_laps):
        return best
    lens = _lens_from_pit_laps(pit_laps, n_laps)

    def consider(lens_arr: np.ndarray, modes: tuple[str, ...], tires: tuple[bool, ...]) -> None:
        total, ok, tail1, tail2 = _shape_times(model, lens_arr, modes, tires)
        if seconds_left is not None:
            ok = ok & _timed_ok(model, seconds_left, total, tail1, tail2, lens_arr[:, -1])
        if not ok.any():
            return
        masked = np.where(ok, total, np.inf)
        i = int(np.argmin(masked))
        key = (stops, sum(tires), n_laps)
        if key not in best or masked[i] < best[key][0]:
            best[key] = (float(masked[i]),
                         PlanSpec(lens=tuple(int(x) for x in lens_arr[i]), modes=modes,
                                  tires=tires))

    coarse_best: list[tuple[float, tuple[str, ...], tuple[bool, ...], np.ndarray]] = []
    for tires in patterns:
        for modes in mode_sets:
            if step == 1:
                consider(lens, modes, tires)
                continue
            total, ok, tail1, tail2 = _shape_times(model, lens, modes, tires)
            if seconds_left is not None:
                # The lap count may shift by one inside the refinement window.
                ok = ok & (np.abs(total - seconds_left) < 3 * model.base)
            if ok.any():
                masked = np.where(ok, total, np.inf)
                i = int(np.argmin(masked))
                coarse_best.append((float(masked[i]), modes, tires, pit_laps[i]))
    if step > 1:
        # Refine the most promising shapes of every tire pattern on the full grid.
        coarse_best.sort(key=lambda item: item[0])
        per_pattern: dict[tuple[bool, ...], int] = {}
        for _, modes, tires, centre in coarse_best:
            if per_pattern.get(tires, 0) >= 12:
                continue
            per_pattern[tires] = per_pattern.get(tires, 0) + 1
            windows = [(int(c) - step + 1, int(c) + step - 1) for c in centre]
            fine = _pit_lap_rows(n_laps, stops, caps, windows=windows)
            if len(fine):
                consider(_lens_from_pit_laps(fine, n_laps), modes, tires)
    return best


def _basis(fit: Fit) -> str:
    per_mode = ", ".join(f"{m.name} {m.n_laps}" for m in
                         sorted(fit.modes.values(), key=lambda m: (m.name != PUSH, m.name)))
    stops = fit.pit.n_stops if fit.pit else 0
    return (f"{sum(m.n_laps for m in fit.modes.values())} clean laps ({per_mode}) and "
            f"{stops} timed pit stop{'s' if stops != 1 else ''}")


def _remaining(race: Race, state: State) -> tuple[int | None, float | None]:
    if race.timed:
        return None, race.minutes * 60.0 - state.elapsed_s
    return race.laps - state.laps_done, None


def solve(race: Race, fit: Fit, state: State = State()) -> Result:
    """Rank the viable plans for a race from the given state."""
    result = Result(kind="no-plan", basis=_basis(fit) if fit.modes else "")
    if not fit.has_lap_model or PUSH not in fit.modes:
        result.reason = "No lap-time model yet: drive a push stint of at least 8 clean laps."
        return result
    model = _Model(fit, race, state)
    notes = result.notes
    if fit.pit is not None and fit.pit.service_together != race.service_together:
        _note(notes, "The pit-stop parameters were fitted under the other fuel-and-tires rule. "
                     "Run the fit again with the same rule as the race.")
    laps_left, seconds_left = _remaining(race, state)
    if (laps_left is not None and laps_left < 1) or (seconds_left is not None and seconds_left <= 0):
        result.reason = "The race is over."
        return result

    stops_owed = max(0, race.mandatory_stops - state.stops_done)
    stop_limit = max(0, race.max_stops - state.stops_done)
    if stops_owed > stop_limit:
        result.reason = "The mandatory stops exceed the stop limit."
        return result
    low_burn, push_burn = min(model.fuel.values()), model.fuel[PUSH]

    def stop_range(n_laps: int) -> range:
        fewest = model.min_stops(n_laps, low_burn, stop_limit)
        if fewest is None:
            return range(0)
        on_push = model.min_stops(n_laps, push_burn, stop_limit)
        lo = max(stops_owed, fewest)
        hi = min(stop_limit, max(lo, on_push if on_push is not None else lo) + 1)
        return range(lo, hi + 1)

    families: dict[tuple, tuple[float, PlanSpec]] = {}
    pit_needed = False
    if seconds_left is None:
        searches = [(laps_left, s) for s in stop_range(laps_left)]
    else:
        searches = []
        rough = int(seconds_left / model.base)
        for s in sorted(set(itertools.chain.from_iterable(
                stop_range(n) for n in range(max(1, rough - 6), rough + 3)))):
            patterns = _tire_patterns(model, s, []) or [(True,) * s]
            if s > 0 and model.pit_fixed is None:
                pit_needed = True
                continue
            for n in _timed_lap_counts(model, s, patterns, seconds_left):
                if s in stop_range(n):
                    searches.append((n, s))
    for n_laps, s in searches:
        if s > 0 and (model.pit_fixed is None or model.rate is None):
            pit_needed = True
            continue
        patterns = _tire_patterns(model, s, notes)
        families.update(_search(model, n_laps, s, patterns, seconds_left, notes))

    if not families:
        if pit_needed:
            result.reason = ("This race needs a pit stop and none has been timed yet: log one "
                             "stop with fuel, in and out, in practice.")
        elif notes:
            result.reason = notes.pop(0)
        else:
            result.reason = ("No plan fits the tank and the rules with the pace modes "
                             "measured so far.")
        return result

    # Exact evaluation of every family's best plan.
    plans = [_evaluate(model, spec) for _, spec in families.values()]
    plans = [p for p in plans if p.feasible]
    if not plans:
        result.reason = "No plan fits the tank and the rules with the pace modes measured so far."
        return result

    # Timed races: plans may run a different number of laps. Rank them by the
    # time each needs for the laps they all complete.
    common = min(p.spec.laps for p in plans)
    for i, plan in enumerate(plans):
        if plan.spec.laps > common:
            cut = _evaluate(model, plan.spec, upto=common)
            plan.metric_s, plan.vec, plan.exposure = cut.time_s, cut.vec, cut.exposure
    by_family: dict[tuple[int, int], Plan] = {}
    for plan in plans:
        key = (plan.spec.stops, sum(plan.spec.tires))
        if key not in by_family or plan.metric_s < by_family[key].metric_s:
            by_family[key] = plan
    ranked = sorted(by_family.values(), key=lambda p: p.metric_s)
    best = ranked[0]
    result.race_laps = best.laps

    # A race that needs no stop on push gets a start-fuel number and nothing else.
    if stops_owed == 0 and best.spec.stops == 0 and set(best.spec.modes) == {PUSH} \
            and not model.live:
        result.kind = "start-fuel"
        result.start_fuel_l = best.start_fuel_l
        best.gap_u_s = _gap_uncertainty(model, best.vec, None)
        result.plans = [best]
        notes.clear()
        return result

    best.gap_u_s = _gap_uncertainty(model, best.vec, None)
    for plan in ranked[1:]:
        plan.gap_s = plan.metric_s - best.metric_s
        plan.gap_u_s = _gap_uncertainty(model, plan.vec, best.vec)
    if not fit.fuel_weight_fitted:
        for plan in ranked[1:]:
            lighter = plan.exposure - best.exposure     # litre-laps lighter than measured
            if lighter > 1e-6 and plan.gap_s > 0:
                plan.flip_weight = model.weight + plan.gap_s / lighter

    # Viable plans. A plan stays on the list when it could still be the best
    # (inside three U, or the ranking turns on fuel weight that was never
    # measured), or when it is the answer to "drop a stop or take one": the
    # best plan with one stop fewer or more, if it costs less than a stop.
    best_of_stops: dict[int, Plan] = {}
    for plan in ranked:
        best_of_stops.setdefault(plan.spec.stops, plan)

    def viable(plan: Plan) -> bool:
        if plan.gap_s <= 3 * plan.gap_u_s:
            return True
        if abs(plan.spec.stops - best.spec.stops) > 1:
            return False
        if plan.flip_weight is not None and plan.flip_weight < PLAUSIBLE_FUEL_WEIGHT:
            return True
        return (best_of_stops[plan.spec.stops] is plan and plan.spec.stops != best.spec.stops
                and plan.gap_s <= (model.pit_fixed or 0.0))

    shown = [best] + [p for p in ranked[1:] if viable(p)][:MAX_PLANS - 1]
    close = [p for p in shown[1:] if p.gap_s <= TOO_CLOSE_SIGMAS * p.gap_u_s]
    for plan in close:
        plan.verdict = "too close to call"
    if len(ranked) == 1:
        best.verdict = "only plan that works"
    elif close:
        best.verdict = "fastest on paper, too close to call"
    else:
        best.verdict = "clear winner"

    for plan in shown[1:]:
        if plan.flip_weight is not None and plan.flip_weight < 2 * PLAUSIBLE_FUEL_WEIGHT:
            plan.warnings.append(
                f"it beats plan 1 if fuel weight is worth more than "
                f"{plan.flip_weight:.3f} s per litre, which is not measured")
    flippable = [p for p in shown[1:]
                 if p.flip_weight is not None and p.flip_weight < PLAUSIBLE_FUEL_WEIGHT]
    if flippable:
        # The ranking can flip on something that was never measured: no winner yet.
        for plan in flippable:
            plan.verdict = "depends on fuel weight, not measured"
        best.verdict = "fastest on paper, not settled"
        rival = min(flippable, key=lambda p: p.flip_weight)
        result.open_decisions.append(
            f"'{best.title}' against '{rival.title}' turns on the fuel-weight effect, "
            f"which is not measured. Drive one push stint from a different fuel load "
            f"to settle it.")
    result.kind = "plans"
    result.plans = shown
    return result


# -- adjusting a plan ------------------------------------------------------------------

def rescore(race: Race, fit: Fit, plan: Plan, *, state: State = State(),
            pit_laps: dict[int, int] | None = None, fuel: dict[int, float] | None = None,
            modes: dict[int, str] | None = None, tires: dict[int, bool] | None = None,
            reference: Plan | None = None) -> Plan:
    """Apply adjustments to a plan and score the result.

    Stops and stints are numbered from 1; fuel[0] is the start fuel. Only what
    is named changes: other pit laps stay put, and fuel amounts that are not
    set are recomputed as just enough. In a timed race the lap count is
    recomputed. The gap is measured against `reference` (normally plan 1).
    """
    model = _Model(fit, race, state)
    spec = plan.spec
    stops_n = spec.stops
    problems: list[str] = []

    new_modes = list(spec.modes)
    for index, mode in (modes or {}).items():
        if not 1 <= index <= stops_n + 1:
            problems.append(f"there is no stint {index}")
        elif mode not in model.fuel:
            problems.append(f"pace mode {mode!r} has not been measured; choose from "
                            f"{', '.join(model.modes)}")
        else:
            new_modes[index - 1] = mode
    new_tires = list(spec.tires)
    for index, change in (tires or {}).items():
        if not 1 <= index <= stops_n:
            problems.append(f"there is no stop {index}")
        else:
            new_tires[index - 1] = change
    laps_at = [lap - state.laps_done for lap in
               (s.lap for s in plan.stops)]            # pit laps counted from now
    for index, lap in (pit_laps or {}).items():
        if not 1 <= index <= stops_n:
            problems.append(f"there is no stop {index}")
        else:
            laps_at[index - 1] = lap - state.laps_done
    adds: list[float | None] = list(spec.fuel_adds) if spec.fuel_adds else [None] * stops_n
    start_fuel = spec.start_fuel
    for index, litres in (fuel or {}).items():
        if index == 0:
            start_fuel = litres
        elif not 1 <= index <= stops_n:
            problems.append(f"there is no stop {index}")
        else:
            adds[index - 1] = litres

    total_laps = spec.laps
    if any(b <= a for a, b in zip([0] + laps_at, laps_at)) or (laps_at and laps_at[-1] >= total_laps
                                                              and not race.timed):
        problems.append("pit laps must be in order and before the last lap")
    if problems:
        return replace(plan, problems=problems, verdict="", gap_s=0.0, gap_u_s=0.0)

    def build(n_laps: int) -> PlanSpec:
        lens = tuple(np.diff([0] + laps_at + [n_laps]).tolist())
        return PlanSpec(lens=lens, modes=tuple(new_modes), tires=tuple(new_tires),
                        fuel_adds=tuple(adds), start_fuel=start_fuel)

    if race.timed:
        seconds_left = race.minutes * 60.0 - state.elapsed_s
        first = (laps_at[-1] if laps_at else 0) + (2 if race.white_flag_lap else 1)
        chosen = None
        for n_laps in range(first, total_laps + 40):
            candidate = _evaluate(model, build(n_laps))
            before = _evaluate(model, candidate.spec,
                               upto=n_laps - (2 if race.white_flag_lap else 1)).time_s
            ends = _evaluate(model, candidate.spec, upto=n_laps - 1).time_s \
                if race.white_flag_lap else candidate.time_s
            if before < seconds_left <= ends:
                chosen = candidate
                break
        if chosen is None:
            return replace(plan, problems=["no lap count makes this plan end with the clock"],
                           verdict="", gap_s=0.0, gap_u_s=0.0)
        new = chosen
    else:
        new = _evaluate(model, build(total_laps))

    if reference is not None and new.feasible:
        common = min(new.spec.laps, reference.spec.laps)
        mine = _evaluate(model, new.spec, upto=common) if new.spec.laps > common else new
        theirs = (_evaluate(model, reference.spec, upto=common)
                  if reference.spec.laps > common else _evaluate(model, reference.spec))
        new.metric_s = mine.time_s
        new.gap_s = mine.time_s - theirs.time_s
        new.gap_u_s = _gap_uncertainty(model, mine.vec, theirs.vec)
        if abs(new.gap_s) <= TOO_CLOSE_SIGMAS * new.gap_u_s:
            new.verdict = "too close to call"
    return new
