"""Fitter: turns tagged stints into parameters, each with a spread and a lap count.

The lap-time model is linear:

    lap time = base + cost[mode] + tire wear so far + fuel_weight * (fuel on board - ref)

where tire wear grows by `tire_age_loss` for every lap driven on the set. The
fuel-weight term is fitted only when stints at two different fuel loads exist;
otherwise it is declined and `tire_age_loss` is the net slope of a stint
(tire wear minus fuel burn-off).

A parameter the stints cannot support is declined, never guessed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logger import TAGS, mode_name

MIN_MODE_LAPS = 8           # clean laps a pace mode needs before it is offered
MIN_LOAD_DIFF_FRAC = 0.20   # fuel loads must differ by this share of the tank
MIN_LOAD_STINT_LAPS = 4     # clean laps a stint needs to count as a fuel load
OUTLIER_SIGMAS = 5.0        # laps further than this from the fit are dropped once
MIN_FUEL_STOP_L = 2.0       # a stop needs this much fuel to measure the refuel rate
PUSH = "push"
DEFAULT_GROUP = "default"


@dataclass
class Param:
    name: str
    mode: str | None
    value: float | None
    spread: float | None        # standard error of the value
    n: int                      # laps (or stops) behind the value
    unit: str
    status: str = "ok"          # 'ok' or 'declined'
    note: str = ""


@dataclass
class ModeFit:
    name: str
    tag: str
    target: float | None
    fuel_per_lap: float
    fuel_sd: float              # lap-to-lap spread, litres
    n_laps: int
    max_age: int                # oldest tires this mode was measured on, in laps
    slope_group: str            # which tire_age_loss this mode wears the tires at


@dataclass
class PitFit:
    n_stops: int
    fixed_loss: float                   # in-lap + out-lap loss without the standstill
    fixed_loss_se: float
    lane_loss: float | None             # the part explained by the pit-lane speed limit
    out_lap_penalty: float | None       # the rest: braking, pulling away, cold tires
    refuel_rate: float | None           # litres per second
    refuel_rate_se: float | None
    overhead: float                     # standstill that is neither fuel nor tires
    overhead_note: str
    tire_time: float | None
    tire_time_se: float | None
    tire_time_kind: str | None          # 'measured' or 'upper-bound'
    service_together: bool


@dataclass
class Fit:
    car: str
    track: str
    session_ids: list[int]
    tank_l: float | None
    params: list[Param] = field(default_factory=list)
    modes: dict[str, ModeFit] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    theta: list[float] = field(default_factory=list)
    cov: list[list[float]] = field(default_factory=list)
    resid_sd: float | None = None
    ref_fuel: float | None = None       # fuel on board that `base` refers to
    fuel_weight_fitted: bool = False
    pit: PitFit | None = None
    n_clean_laps: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_lap_model(self) -> bool:
        return bool(self.columns)

    def coef(self, column: str) -> float:
        return self.theta[self.columns.index(column)] if column in self.columns else 0.0

    def param(self, name: str, mode: str | None = None) -> Param | None:
        for p in self.params:
            if p.name == name and p.mode == mode:
                return p
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fit":
        data = dict(data)
        data["params"] = [Param(**p) for p in data.get("params", [])]
        data["modes"] = {k: ModeFit(**v) for k, v in data.get("modes", {}).items()}
        data["pit"] = PitFit(**data["pit"]) if data.get("pit") else None
        return cls(**data)


# -- preparing the laps -----------------------------------------------------------

def _lap_mode(tag: str, target: Any) -> str | None:
    if tag not in TAGS:
        return None
    return mode_name(tag, None if target is None or pd.isna(target) else float(target))


def prepare_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Add the pace mode and a usable flag to the raw lap rows."""
    df = laps.sort_values(["session_id", "lap_number"]).reset_index(drop=True).copy()
    df["mode"] = [_lap_mode(t, g) for t, g in zip(df["stint_tag"], df["tag_target"])]
    df["counts"] = df["lap_time_s"].notna() & ~df["unclean_reason"].str.contains("partial")
    df["usable"] = ((df["clean"] == 1) & (df["traffic"] == 0) & (df["wet"] == 0)
                    & df["mode"].notna() & df["lap_time_s"].notna()
                    & (df["fuel_used_l"] > 0))
    return df


def _add_wear_counts(df: pd.DataFrame, group_of: dict[str, str]) -> list[str]:
    """Count, for every lap, the laps already driven on its tire set per wear group."""
    groups = sorted(set(group_of.values()) | {DEFAULT_GROUP})
    counts_by_group: dict[str, list[int]] = {g: [] for g in groups}
    running: dict[str, int] = {}
    prev_session = prev_stint = prev_age = None
    for session, stint, age, mode, counts in zip(
            df["session_id"], df["stint_number"], df["tire_age"], df["mode"], df["counts"]):
        new_set = (session != prev_session or age < (prev_age or 0)
                   or (age == 0 and stint != prev_stint))
        if new_set:
            running = {g: 0 for g in groups}
        known = sum(running.values())
        if age > known:
            running[DEFAULT_GROUP] += age - known      # laps we did not see
        for g in groups:
            counts_by_group[g].append(running[g])
        if counts:
            running[group_of.get(mode, DEFAULT_GROUP)] += 1
        prev_session, prev_stint, prev_age = session, stint, age
    for g in groups:
        df[f"wear:{g}"] = counts_by_group[g]
    return groups


# -- least squares ---------------------------------------------------------------

def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
    n, p = x.shape
    if n <= p or np.linalg.matrix_rank(x) < p:
        return None
    theta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ theta
    sigma2 = float(resid @ resid) / (n - p)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    return theta, cov, math.sqrt(sigma2), resid


def _robust_ols(x: np.ndarray, y: np.ndarray):
    """Least squares with one pass of outlier rejection. Returns (fit, kept mask)."""
    keep = np.ones(len(y), dtype=bool)
    fit = _ols(x, y)
    if fit is None:
        return None, keep
    resid = fit[3]
    centre = np.median(resid)
    robust_sd = 1.4826 * np.median(np.abs(resid - centre))
    if robust_sd > 0:
        keep = np.abs(resid - centre) <= OUTLIER_SIGMAS * robust_sd
        if not keep.all():
            refit = _ols(x[keep], y[keep])
            if refit is not None:
                return refit, keep
            keep = np.ones(len(y), dtype=bool)
    return fit, keep


# -- the fit ----------------------------------------------------------------------

def fit_parameters(laps: pd.DataFrame, pits: pd.DataFrame, *, car: str, track: str,
                   session_ids: list[int] | None = None, tank_l: float | None = None,
                   service_together: bool = True) -> Fit:
    """Fit every parameter the logged stints support."""
    fit = Fit(car=car, track=track, tank_l=tank_l,
              session_ids=sorted(set(session_ids if session_ids is not None
                                     else laps["session_id"].tolist())))
    df = prepare_laps(laps)
    usable = df[df["usable"]]
    fit.n_clean_laps = int(len(usable))

    # Fuel per lap and which pace modes have enough laps to be offered.
    for mode, rows in usable.groupby("mode", sort=True):
        n = len(rows)
        if n < MIN_MODE_LAPS:
            fit.params.append(Param("fuel_per_lap", mode, None, None, n, "L/lap", "declined",
                                    f"only {n} clean laps; a pace mode needs {MIN_MODE_LAPS}"))
            continue
        tag = rows["stint_tag"].iloc[0]
        target = rows["tag_target"].iloc[0]
        sd = float(rows["fuel_used_l"].std(ddof=1))
        fit.modes[mode] = ModeFit(
            name=mode, tag=tag, target=None if pd.isna(target) else float(target),
            fuel_per_lap=float(rows["fuel_used_l"].mean()), fuel_sd=sd, n_laps=n,
            max_age=int(rows["tire_age"].max()),
            slope_group=mode if tag == "tire-save" else DEFAULT_GROUP)
        fit.params.append(Param("fuel_per_lap", mode, fit.modes[mode].fuel_per_lap,
                                sd / math.sqrt(n), n, "L/lap",
                                note=f"lap-to-lap spread {sd:.3f} L"))

    if PUSH not in fit.modes:
        fit.params.append(Param("base_lap_time", None, None, None, 0, "s", "declined",
                                f"no push stint with {MIN_MODE_LAPS} clean laps yet"))
        fit.notes.append("Drive a push stint first: every other number is measured against it.")
    else:
        _fit_lap_model(fit, df)

    _fit_pit(fit, df, pits, service_together)
    return fit


def _fit_lap_model(fit: Fit, df: pd.DataFrame) -> None:
    group_of = {m.name: m.slope_group for m in fit.modes.values()}
    groups = _add_wear_counts(df, group_of)
    rows = df[df["usable"] & df["mode"].isin(fit.modes)].copy()
    fuel_rate = rows["mode"].map({m.name: m.fuel_per_lap for m in fit.modes.values()})
    # The fuel load a lap's stint would have had on new tires.
    rows["load"] = rows["fuel_start_l"] + fuel_rate * rows["tire_age"]
    stint_loads = (rows.groupby(["session_id", "stint_number"])["load"]
                   .agg(["median", "size"]))
    stint_loads = stint_loads[stint_loads["size"] >= MIN_LOAD_STINT_LAPS]["median"]
    fit.ref_fuel = float(rows["load"].mean())

    used_groups = groups
    cost_modes = [m for m in sorted(fit.modes) if m != PUSH]
    columns = ["base"] + [f"wear:{g}" for g in used_groups] + [f"cost:{m}" for m in cost_modes]
    y = rows["lap_time_s"].to_numpy(dtype=float)

    def design(cols: list[str]) -> np.ndarray:
        parts = []
        for col in cols:
            if col == "base":
                parts.append(np.ones(len(rows)))
            elif col == "fuel":
                parts.append((rows["fuel_start_l"] - fit.ref_fuel).to_numpy(dtype=float))
            elif col.startswith("cost:"):
                parts.append((rows["mode"] == col[5:]).to_numpy(dtype=float))
            else:
                parts.append(rows[col].to_numpy(dtype=float))
        return np.column_stack(parts)

    # Fuel weight: only when two different fuel loads were driven.
    tank = fit.tank_l or float(rows["load"].max())
    spread_l = float(stint_loads.max() - stint_loads.min()) if len(stint_loads) else 0.0
    need_l = MIN_LOAD_DIFF_FRAC * tank
    weight_note = ""
    result = None
    if spread_l >= need_l:
        with_fuel = columns + ["fuel"]
        result, keep = _robust_ols(design(with_fuel), y)
        if result is None:
            weight_note = "the stints do not separate fuel weight from tire wear"
        else:
            idx = with_fuel.index("fuel")
            weight, weight_se = float(result[0][idx]), math.sqrt(result[1][idx, idx])
            if weight > 2 * weight_se:
                columns = with_fuel
                fit.fuel_weight_fitted = True
                fit.params.append(Param("fuel_weight", None, weight, weight_se,
                                        int(keep.sum()), "s/L",
                                        note=f"fuel loads {spread_l:.0f} L apart"))
            else:
                weight_note = (f"not distinguishable from zero in these stints "
                               f"({weight:+.4f} ± {weight_se:.4f} s/L)")
                result = None
    else:
        low = max(0.0, float(stint_loads.max()) - 2 * need_l) if len(stint_loads) else 0.0
        weight_note = (f"only one fuel load driven (stints within {spread_l:.0f} L of each "
                       f"other); drive a push stint starting from about {low:.0f} L")
    if not fit.fuel_weight_fitted:
        fit.params.append(Param("fuel_weight", None, None, None, len(rows), "s/L",
                                "declined", weight_note))
        result, keep = _robust_ols(design(columns), y)

    if result is None:
        fit.params.append(Param("base_lap_time", None, None, None, len(rows), "s", "declined",
                                "too few clean laps to fit the lap-time model"))
        fit.modes = {}
        return
    theta, cov, resid_sd, _ = result
    n = int(keep.sum())
    dropped = len(rows) - n
    if dropped:
        fit.notes.append(f"{dropped} lap(s) far from the rest were left out of the lap-time fit.")
    fit.columns = columns
    fit.theta = [float(v) for v in theta]
    fit.cov = [[float(v) for v in row] for row in cov]
    fit.resid_sd = resid_sd

    def se(col: str) -> float:
        i = columns.index(col)
        return math.sqrt(max(0.0, cov[i, i]))

    fit.params.append(Param("base_lap_time", None, fit.coef("base"), se("base"), n, "s",
                            note=f"push, new tires, {fit.ref_fuel:.0f} L on board; "
                                 f"lap-to-lap scatter {resid_sd:.3f} s"))
    net = "" if fit.fuel_weight_fitted else "net of fuel burn-off"
    for g in used_groups:
        col = f"wear:{g}"
        fit.params.append(Param("tire_age_loss", None if g == DEFAULT_GROUP else g,
                                fit.coef(col), se(col), n, "s/lap", note=net))
    for m in cost_modes:
        col = f"cost:{m}"
        fit.params.append(Param("mode_cost", m, fit.coef(col), se(col),
                                fit.modes[m].n_laps, "s/lap"))


def predict_lap(fit: Fit, *, mode: str, wear: dict[str, float], fuel_l: float) -> float:
    """Lap time the model expects for one lap."""
    t = fit.coef("base")
    if mode != PUSH:
        t += fit.coef(f"cost:{mode}")
    for group, laps in wear.items():
        t += fit.coef(f"wear:{group}") * laps
    if fit.fuel_weight_fitted and fit.ref_fuel is not None:
        t += fit.coef("fuel") * (fuel_l - fit.ref_fuel)
    return t


def _fit_pit(fit: Fit, df: pd.DataFrame, pits: pd.DataFrame, service_together: bool) -> None:
    stops = pits[pits["pit_lane_time_s"].notna() & (pits["stationary_time_s"] > 0)]
    if stops.empty:
        for name, unit in (("pit_lane_loss", "s"), ("out_lap_penalty", "s"),
                           ("refuel_rate", "L/s"), ("tire_change_time", "s")):
            fit.params.append(Param(name, None, None, None, 0, unit, "declined",
                                    "no timed pit stop logged"))
        return

    fixed, lane_parts, ref_scatter = [], [], []
    by_session = {sid: rows.set_index("lap_number") for sid, rows in df.groupby("session_id")}
    wear_cols = [c for c in df.columns if c.startswith("wear:")]
    for stop in stops.itertuples():
        laps = by_session.get(stop.session_id)
        if laps is None or stop.lap not in laps.index:
            continue
        numbers = [stop.lap] if laps.loc[stop.lap, "pit_road"] == "in+out" \
            else [stop.lap, stop.lap + 1]
        if any(n not in laps.index for n in numbers):
            continue
        excursion = laps.loc[numbers]
        if excursion["lap_time_s"].isna().any() or (excursion["incidents"] > 0).any() \
                or excursion["unclean_reason"].str.contains("partial|flag").any():
            continue
        near = laps[(laps.index >= stop.lap - 6) & (laps.index <= stop.lap + 7)
                    & (laps["clean"] == 1) & (laps["traffic"] == 0)]
        expected = []
        for _, row in excursion.iterrows():
            if fit.has_lap_model and row["mode"] in fit.modes:
                wear = {c[5:]: row[c] for c in wear_cols}
                expected.append(predict_lap(fit, mode=row["mode"], wear=wear,
                                            fuel_l=row["fuel_start_l"]))
            elif len(near) >= 2:
                expected.append(float(near["lap_time_s"].median()))
        if len(expected) != len(numbers):
            continue
        total = float(excursion["lap_time_s"].sum()) - sum(expected)
        fixed.append(total - stop.stationary_time_s)
        if len(near) >= 2:
            ref_scatter.append(float(near["lap_time_s"].std(ddof=1)))
        if not (pd.isna(stop.entry_pct) or pd.isna(stop.exit_pct)):
            share = (stop.exit_pct - stop.entry_pct) % 1.0
            reference = sum(expected) / len(expected)
            lane_parts.append((stop.pit_lane_time_s - stop.stationary_time_s)
                              - share * reference)

    fuel_stops = stops[(stops["fuel_added_l"] >= MIN_FUEL_STOP_L) & (stops["fuel_time_s"] >= 1.0)]
    rates = (fuel_stops["fuel_added_l"] / fuel_stops["fuel_time_s"]).to_numpy()
    rate = float(rates.mean()) if len(rates) else None
    rate_se = float(rates.std(ddof=1) / math.sqrt(len(rates))) if len(rates) >= 2 else None
    if rate is None:
        fit.params.append(Param("refuel_rate", None, None, None, 0, "L/s", "declined",
                                "no stop that took on fuel"))
    else:
        fit.params.append(Param("refuel_rate", None, rate, rate_se, len(rates), "L/s",
                                note="" if rate_se is not None else "one stop: spread unknown"))

    # Standstill that is neither fuelling nor tires.
    fuel_only = fuel_stops[fuel_stops["tires_changed"] == 0]
    delays = fuel_stops["fuel_delay_s"].dropna()
    if len(fuel_only):
        overhead = float((fuel_only["stationary_time_s"] - fuel_only["fuel_time_s"]).mean())
        overhead_note = f"from {len(fuel_only)} fuel-only stop(s)"
    elif service_together and len(delays):
        overhead = float(delays.mean())
        overhead_note = "from the delay before fuel flowed; no fuel-only stop logged"
    else:
        overhead = 0.0
        overhead_note = "not measured; taken as zero"
    overhead = max(0.0, overhead)

    tire_stops = stops[stops["tires_changed"] == 1]
    tire_time = tire_se = None
    tire_kind = None
    if tire_stops.empty:
        fit.params.append(Param("tire_change_time", None, None, None, 0, "s", "declined",
                                "no stop with a tire change logged"))
    else:
        standstill = tire_stops["stationary_time_s"].to_numpy()
        fuelling = tire_stops["fuel_time_s"].to_numpy()
        if service_together:
            measured = standstill[standstill > overhead + fuelling + 1.0]
            if len(measured):
                samples, tire_kind = measured, "measured"
                note = "tires took longer than the fuel"
            else:
                samples, tire_kind = np.array([standstill.min()]), "upper-bound"
                note = ("the tire change finished within the fuelling time, so this is an "
                        "upper bound; a stop with less fuel would measure it")
        else:
            samples, tire_kind = standstill - fuelling - overhead, "measured"
            note = "standstill minus fuelling time"
        tire_time = float(samples.mean())
        tire_se = float(samples.std(ddof=1) / math.sqrt(len(samples))) if len(samples) >= 2 \
            else None
        fit.params.append(Param("tire_change_time", None, tire_time, tire_se, len(samples), "s",
                                note=note))

    if not fixed:
        for name in ("pit_lane_loss", "out_lap_penalty"):
            fit.params.append(Param(name, None, None, None, 0, "s", "declined",
                                    "no stop with a usable in-lap and out-lap"))
        return
    n = len(fixed)
    fixed_loss = float(np.mean(fixed))
    scatter = fit.resid_sd if fit.resid_sd is not None else (
        float(np.mean(ref_scatter)) if ref_scatter else 0.5)
    # Two laps of scatter go into every measured stop.
    floor = scatter * math.sqrt(2.0 / n)
    fixed_se = max(floor, float(np.std(fixed, ddof=1) / math.sqrt(n)) if n >= 2 else 0.0)
    lane_loss = float(np.mean(lane_parts)) if len(lane_parts) == n else None
    out_lap = fixed_loss - lane_loss if lane_loss is not None else None
    fit.pit = PitFit(n_stops=n, fixed_loss=fixed_loss, fixed_loss_se=fixed_se,
                     lane_loss=lane_loss, out_lap_penalty=out_lap, refuel_rate=rate,
                     refuel_rate_se=rate_se, overhead=overhead, overhead_note=overhead_note,
                     tire_time=tire_time, tire_time_se=tire_se, tire_time_kind=tire_kind,
                     service_together=service_together)
    if lane_loss is None:
        fit.params.append(Param("pit_lane_loss", None, fixed_loss, fixed_se, n, "s",
                                note="whole in-lap and out-lap loss, standstill excluded"))
        fit.params.append(Param("out_lap_penalty", None, None, None, n, "s", "declined",
                                "pit entry and exit positions were not recorded"))
    else:
        lane_se = float(np.std(lane_parts, ddof=1) / math.sqrt(n)) if n >= 2 else None
        fit.params.append(Param("pit_lane_loss", None, lane_loss, lane_se, n, "s",
                                note="time lost to the pit-lane speed limit"))
        fit.params.append(Param("out_lap_penalty", None, out_lap, fixed_se, n, "s",
                                note="braking in, pulling away and the cold-tire out-lap"))
    fit.params.append(Param("service_overhead", None, overhead, None, n, "s",
                            note=overhead_note))


# -- comparing two fits -------------------------------------------------------------

@dataclass
class Comparison:
    name: str
    mode: str | None
    a: float
    b: float
    spread: float | None        # combined spread of the two values
    agree: bool | None          # None when a spread is unknown


def compare_fits(a: Fit, b: Fit, sigmas: float = 2.0) -> list[Comparison]:
    """Do two fits of the same car and track agree within their own spread?"""
    out: list[Comparison] = []
    for pa in a.params:
        pb = b.param(pa.name, pa.mode)
        if pb is None or pa.status != "ok" or pb.status != "ok" \
                or pa.value is None or pb.value is None:
            continue
        if pa.spread is None or pb.spread is None:
            out.append(Comparison(pa.name, pa.mode, pa.value, pb.value, None, None))
            continue
        combined = math.hypot(pa.spread, pb.spread)
        out.append(Comparison(pa.name, pa.mode, pa.value, pb.value, combined,
                              abs(pa.value - pb.value) <= sigmas * combined))
    return out
