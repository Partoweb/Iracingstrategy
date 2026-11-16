"""Stint logger: turns telemetry ticks into one row per lap and one per pit stop.

`LapLogger.process()` is a pure state machine over `Sample`s. `run_logger()`
wraps it in the polling loop used against the live sim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .store import Store
from .telemetry import (TIRE_CHANGE_BITS, UNCLEAN_FLAGS, Sample, SessionInfo)

TAGS = ("push", "fuel-save", "tire-save")
UNTAGGED = "untagged"

FUEL_EPS = 0.002            # litres per tick; anything above is fuel going in
STOPPED_SPEED = 0.5         # m/s
LAP_TIME_WAIT = 3.0         # seconds to wait for iRacing's own lap time
LAP_TIME_TOLERANCE = 1.0    # iRacing's time must be this close to our own timer


def mode_name(tag: str, target: float | None) -> str:
    """Pace-mode key used by the fitter and the solver, e.g. 'fuel-save@2.6'."""
    return f"{tag}@{target:g}" if target is not None else tag


@dataclass
class _Lap:
    number: int
    start_time: float
    start_fuel: float
    incidents_start: int
    started_on_pit: bool
    partial: bool               # did not start at the line
    stint: int
    lap_in_stint: int
    tire_age: int
    tag: str
    target: float | None
    fuel_added: float = 0.0
    flags: int = 0
    wet: bool = False
    pit_entered: bool = False
    pit_exited: bool = False
    tag_changed: bool = False
    traffic: bool = False


@dataclass
class _Pit:
    lap: int
    entry_time: float
    entry_pct: float
    tires_used_at_entry: int | None
    stationary: float = 0.0
    stop_start: float | None = None
    first_stop: float | None = None
    fuel_added: float = 0.0
    fuel_first: float | None = None
    fuel_last: float | None = None
    tire_request: bool = False


@dataclass
class _Pending:
    row: dict[str, Any]
    reasons: list[str]
    delta: float
    lap_time_before: float
    cross_time: float
    partial: bool


@dataclass
class LapLogger:
    store: Store
    info_provider: Callable[[int], SessionInfo]
    source: str = "irsdk"
    on_lap: Callable[[dict[str, Any]], None] | None = None
    on_pit: Callable[[dict[str, Any]], None] | None = None
    on_session: Callable[[int, SessionInfo], None] | None = None

    session_id: int | None = None
    tag: str = UNTAGGED
    target: float | None = None

    _info: SessionInfo | None = None
    _session_num: int | None = None
    _prev: Sample | None = None
    _on_track: bool = False
    _cur: _Lap | None = None
    _pit: _Pit | None = None
    _pending: _Pending | None = None
    _abandoned: dict[str, Any] | None = None
    _stint: int = 0
    _new_stint_next: bool = False
    _tire_reset_next: bool = False
    _tires_used_seen: int | None = None
    _runs: int = 0
    last_lap_number: int | None = field(default=None)

    # -- commands ------------------------------------------------------------

    def set_tag(self, tag: str, target: float | None = None) -> None:
        if tag not in TAGS and tag != UNTAGGED:
            raise ValueError(f"unknown stint tag {tag!r}; use one of {', '.join(TAGS)}")
        if (tag, target) == (self.tag, self.target):
            return
        self.tag, self.target = tag, target
        lap, prev = self._cur, self._prev
        if lap is None or not self._on_track or prev is None:
            return
        if prev.on_pit_road and lap.pit_entered:
            return                      # set during a stop: the in-lap keeps its old tag
        lap.tag, lap.target = tag, target
        if not prev.on_pit_road and not lap.partial:
            lap.tag_changed = True      # driven partly under the old tag

    def mark_traffic(self, last: bool = False) -> int | None:
        """Tag the current lap (or the last completed one) as held up by traffic."""
        if last:
            if self.session_id is not None and self.last_lap_number is not None:
                self._flush_pending()
                self.store.mark_traffic(self.session_id, self.last_lap_number)
                return self.last_lap_number
            return None
        if self._cur is not None:
            self._cur.traffic = True
            return self._cur.number
        return None

    # -- the state machine -----------------------------------------------------

    def process(self, s: Sample) -> None:
        if self._session_num != s.session_num:
            self._close_session()
            self._session_num = s.session_num

        self._resolve_pending(s)

        if not s.is_on_track:
            if self._on_track:
                self._leave_track()
            self._prev = s
            return

        if not self._on_track:
            self._enter_track(s)
        else:
            assert self._prev is not None
            self._tick(self._prev, s)
        self._prev = s

    def finalize(self) -> None:
        """Flush what is still open. The lap in progress is not written."""
        self._close_session()

    def _close_session(self) -> None:
        self._flush_pending()
        if self._on_track:
            self._leave_track()
        self.session_id = None
        self._info = None
        self._session_num = None
        self._prev = None
        self._cur = None
        self._abandoned = None
        self._stint = 0
        self._runs = 0
        self._new_stint_next = False
        self._tire_reset_next = False
        self._tires_used_seen = None
        self.last_lap_number = None

    def _leave_track(self) -> None:
        prev = self._prev
        self._on_track = False
        if prev is None:
            return
        if self._pit is not None:
            self._write_pit(prev, exited=False)
        if self._cur is not None:
            self._abandoned = {"lap": self._cur, "fuel": prev.fuel_l, "time": prev.session_time,
                               "incidents": prev.incidents}
        self._cur = None
        self._tires_used_seen = prev.tires_used

    def _enter_track(self, s: Sample) -> None:
        if self.session_id is None:
            info = self.info_provider(s.session_num)
            self._info = info
            self.session_id = self.store.add_session(
                source=self.source, car=info.car, track=info.track,
                session_type=info.session_type, sub_session_id=info.sub_session_id,
                session_num=s.session_num, tank_l=info.tank_l, race_laps=info.race_laps,
                race_seconds=info.race_seconds)
            if self.on_session:
                self.on_session(self.session_id, info)

        abandoned = self._abandoned
        self._abandoned = None
        if abandoned is not None and s.lap > abandoned["lap"].number >= 1:
            # The lap that was under way when the driver left the car never
            # finished. Keep a row for it so lap numbers stay gap-free.
            self._flush_pending()
            lap: _Lap = abandoned["lap"]
            self._write_lap(self._lap_row(lap, end_time=abandoned["time"],
                                          end_fuel=abandoned["fuel"],
                                          end_incidents=abandoned["incidents"], lap_time=None,
                                          source="none"), ["reset"])

        self._on_track = True
        self._runs += 1
        self._stint += 1
        self._new_stint_next = False
        self._tire_reset_next = False
        tire_age = 0
        if self._runs > 1 and abandoned is not None and self._is_race() \
                and s.tires_used is not None and s.tires_used == self._tires_used_seen:
            # Back in the same car on the same set (a tow during a race).
            tire_age = abandoned["lap"].tire_age
        self._cur = _Lap(number=s.lap, start_time=s.session_time, start_fuel=s.fuel_l,
                         incidents_start=s.incidents, started_on_pit=s.on_pit_road,
                         partial=True, stint=self._stint, lap_in_stint=1, tire_age=tire_age,
                         tag=self.tag, target=self.target)

    def _is_race(self) -> bool:
        return self._info is not None and "race" in self._info.session_type.lower()

    def _tick(self, prev: Sample, s: Sample) -> None:
        lap = self._cur
        assert lap is not None
        lap.flags |= s.flags
        lap.wet = lap.wet or s.wet

        if s.on_pit_road and not prev.on_pit_road:
            self._pit = _Pit(lap=lap.number, entry_time=s.session_time,
                             entry_pct=s.lap_dist_pct, tires_used_at_entry=prev.tires_used)
            lap.pit_entered = True
            self._new_stint_next = True

        gained = s.fuel_l - prev.fuel_l
        if gained > FUEL_EPS:
            lap.fuel_added += gained

        if self._pit is not None:
            self._pit_tick(prev, s, gained)

        if prev.on_pit_road and not s.on_pit_road:
            lap.pit_exited = True
            if self._pit is not None:
                self._write_pit(s, exited=True)

        if s.lap > lap.number:
            self._flush_pending()
            self._complete_lap(prev, s)
            self._begin_lap(s, partial=False)
        elif s.lap < lap.number:
            # The lap counter went backwards (session restart): start over.
            self._begin_lap(s, partial=True)

    def _pit_tick(self, prev: Sample, s: Sample, gained: float) -> None:
        pit = self._pit
        assert pit is not None
        in_stall = True if s.in_pit_stall is None else s.in_pit_stall
        stopped = s.on_pit_road and in_stall and s.speed < STOPPED_SPEED
        if stopped:
            if pit.stop_start is None:
                pit.stop_start = s.session_time
                if pit.first_stop is None:
                    pit.first_stop = s.session_time
            if s.pit_sv_flags & TIRE_CHANGE_BITS:
                pit.tire_request = True
        elif pit.stop_start is not None:
            pit.stationary += s.session_time - pit.stop_start
            pit.stop_start = None
        if gained > FUEL_EPS:
            pit.fuel_added += gained
            if pit.fuel_first is None:
                pit.fuel_first = prev.session_time
            pit.fuel_last = s.session_time

    def _write_pit(self, s: Sample, *, exited: bool) -> None:
        pit = self._pit
        assert pit is not None and self.session_id is not None
        self._pit = None
        if pit.stop_start is not None:
            pit.stationary += s.session_time - pit.stop_start
        if pit.tires_used_at_entry is not None and s.tires_used is not None:
            tires = s.tires_used > pit.tires_used_at_entry
        else:
            tires = pit.tire_request and pit.stationary > 0
        fuel_time = 0.0
        fuel_delay = None
        if pit.fuel_first is not None and pit.fuel_last is not None:
            fuel_time = pit.fuel_last - pit.fuel_first
            if pit.first_stop is not None:
                fuel_delay = max(0.0, pit.fuel_first - pit.first_stop)
        row = {
            "session_id": self.session_id, "lap": pit.lap, "entry_time_s": pit.entry_time,
            "pit_lane_time_s": (s.session_time - pit.entry_time) if exited else None,
            "stationary_time_s": pit.stationary, "fuel_time_s": fuel_time,
            "fuel_delay_s": fuel_delay,
            "fuel_added_l": pit.fuel_added, "tires_changed": int(tires),
            "entry_pct": pit.entry_pct, "exit_pct": s.lap_dist_pct if exited else None,
        }
        self.store.add_pit_stop(row)
        if tires and self._cur is not None:
            if self._cur.number == pit.lap:
                self._tire_reset_next = True        # the next lap is the first on the new set
            else:
                self._cur.tire_age = 0              # this lap is the out-lap
        if self.on_pit:
            self.on_pit(row)

    def _begin_lap(self, s: Sample, *, partial: bool) -> None:
        old = self._cur
        # The roll from the stall or the grid to the line is not a lap: it does
        # not advance the stint or the tire age.
        step = 0 if (old is not None and old.partial) else 1
        if self._new_stint_next or old is None:
            self._stint += 1 if old is not None else 0
            lap_in_stint = 1
        else:
            lap_in_stint = old.lap_in_stint + step
        self._new_stint_next = False
        tire_age = 0 if (self._tire_reset_next or old is None) else old.tire_age + step
        self._tire_reset_next = False
        self._cur = _Lap(number=s.lap, start_time=s.session_time, start_fuel=s.fuel_l,
                         incidents_start=s.incidents, started_on_pit=s.on_pit_road,
                         partial=partial, stint=self._stint, lap_in_stint=lap_in_stint,
                         tire_age=tire_age, tag=self.tag, target=self.target)

    def _lap_row(self, lap: _Lap, *, end_time: float, end_fuel: float, end_incidents: int,
                 lap_time: float | None, source: str) -> dict[str, Any]:
        prev = self._prev
        if lap.pit_entered and (lap.started_on_pit or lap.pit_exited):
            pit_road = "in+out"
        elif lap.pit_entered:
            pit_road = "in"
        elif lap.started_on_pit or lap.pit_exited:
            pit_road = "out"
        else:
            pit_road = "none"
        return {
            "session_id": self.session_id, "lap_number": lap.number,
            "lap_time_s": lap_time, "lap_time_source": source, "session_time_s": end_time,
            "fuel_start_l": lap.start_fuel, "fuel_at_line_l": end_fuel,
            "fuel_added_l": lap.fuel_added,
            "fuel_used_l": lap.start_fuel - end_fuel + lap.fuel_added,
            "stint_number": lap.stint, "lap_in_stint": lap.lap_in_stint,
            "tire_age": lap.tire_age, "stint_tag": lap.tag, "tag_target": lap.target,
            "track_temp_c": prev.track_temp if prev else None,
            "air_temp_c": prev.air_temp if prev else None,
            "wet": int(lap.wet), "pit_road": pit_road, "flags": lap.flags,
            "incidents": max(0, end_incidents - lap.incidents_start),
            "traffic": int(lap.traffic),
        }

    def _complete_lap(self, prev: Sample, s: Sample) -> None:
        lap = self._cur
        assert lap is not None
        if lap.number < 1:
            return                  # the run from the grid or the stall to the line
        row = self._lap_row(lap, end_time=s.session_time, end_fuel=s.fuel_l,
                            end_incidents=s.incidents, lap_time=None, source="none")
        reasons: list[str] = []
        if lap.partial:
            reasons.append("partial")
        if row["pit_road"] in ("out", "in+out"):
            reasons.append("out-lap")
        if row["pit_road"] in ("in", "in+out"):
            reasons.append("in-lap")
        if row["incidents"] > 0:
            reasons.append("incident")
        if lap.flags & UNCLEAN_FLAGS:
            reasons.append("flag")
        if lap.tag_changed:
            reasons.append("tag-change")
        if self._is_race() and lap.number == 1:
            reasons.append("race-start")
        self._pending = _Pending(row=row, reasons=reasons,
                                 delta=s.session_time - lap.start_time,
                                 lap_time_before=prev.lap_last_time,
                                 cross_time=s.session_time, partial=lap.partial)
        self._resolve_pending(s)

    def _resolve_pending(self, s: Sample | None, force: bool = False) -> None:
        """Attach a lap time to the lap that just ended, then write it.

        iRacing publishes its own (interpolated) lap time a few ticks after the
        line. We wait for it briefly and fall back to our own timer.
        """
        p = self._pending
        if p is None:
            return
        lap_time: float | None = None
        source = "timer"
        if p.partial:
            lap_time = p.delta
        elif s is not None and s.lap_last_time > 0 \
                and s.lap_last_time != p.lap_time_before \
                and abs(s.lap_last_time - p.delta) < LAP_TIME_TOLERANCE:
            lap_time, source = s.lap_last_time, "sim"
        elif force or s is None or s.session_time - p.cross_time > LAP_TIME_WAIT:
            # Two laps with the very same time look like "no update"; accept
            # iRacing's value when it matches our own timer closely.
            if s is not None and s.lap_last_time > 0 \
                    and abs(s.lap_last_time - p.delta) < 0.25:
                lap_time, source = s.lap_last_time, "sim"
            else:
                lap_time = p.delta
        else:
            return
        self._pending = None
        p.row["lap_time_s"] = lap_time
        p.row["lap_time_source"] = source
        self._write_lap(p.row, p.reasons)

    def _flush_pending(self) -> None:
        self._resolve_pending(self._prev, force=True)

    def _write_lap(self, row: dict[str, Any], reasons: list[str]) -> None:
        row["clean"] = int(not reasons)
        row["unclean_reason"] = ",".join(reasons)
        self.store.add_lap(row)
        self.last_lap_number = row["lap_number"]
        if self.on_lap:
            self.on_lap(row)


# -- checks on a logged session ---------------------------------------------------

def verify_session(store: Store, session_id: int, tolerance_l: float = 0.05) -> list[str]:
    """M1 acceptance checks. Returns a list of problems; empty means all good."""
    laps = store.laps_df([session_id])
    problems: list[str] = []
    if laps.empty:
        return ["no laps logged"]
    numbers = laps["lap_number"].tolist()
    expected = list(range(numbers[0], numbers[-1] + 1))
    missing = sorted(set(expected) - set(numbers))
    if missing:
        problems.append(f"missing laps: {', '.join(map(str, missing))}")
    for stint, rows in laps.groupby("stint_number"):
        used = rows["fuel_used_l"].sum()
        drop = (rows["fuel_start_l"].iloc[0] - rows["fuel_at_line_l"].iloc[-1]
                + rows["fuel_added_l"].sum())
        if abs(used - drop) > tolerance_l:
            problems.append(f"stint {stint}: fuel used per lap sums to {used:.2f} L but the "
                            f"tank dropped {drop:.2f} L")
        starts = rows["fuel_start_l"].to_numpy()[1:]
        ends = rows["fuel_at_line_l"].to_numpy()[:-1]
        consecutive = rows["lap_number"].diff().to_numpy()[1:] == 1
        jumps = abs(starts - ends)[consecutive]
        if len(jumps) and jumps.max() > tolerance_l:
            problems.append(f"stint {stint}: fuel level jumps {jumps.max():.2f} L between laps")
    return problems


# -- live loop ---------------------------------------------------------------------

def run_logger(source: Any, store: Store, *, hz: float = 60.0,
               say: Callable[[str], None] = print,
               poll_inputs: Callable[[], list[tuple[str, dict[str, Any]]]] | None = None,
               on_lap: Callable[[LapLogger, dict[str, Any]], None] | None = None,
               stop: Callable[[], bool] | None = None) -> None:
    """Poll the telemetry source until interrupted and log every lap."""
    logger = LapLogger(store=store, info_provider=source.session_info, source=source.name)

    def lap_line(row: dict[str, Any]) -> None:
        lap_time = "   --   " if row["lap_time_s"] is None else f"{row['lap_time_s']:8.3f}"
        state = "clean" if row["clean"] else row["unclean_reason"]
        say(f"lap {row['lap_number']:>3}  {lap_time} s  fuel {row['fuel_used_l']:5.2f} L "
            f"(left {row['fuel_at_line_l']:5.1f})  stint {row['stint_number']} lap "
            f"{row['lap_in_stint']}  {mode_name(row['stint_tag'], row['tag_target'])}  {state}")
        if on_lap:
            on_lap(logger, row)

    def pit_line(row: dict[str, Any]) -> None:
        lane = row["pit_lane_time_s"]
        say(f"pit stop on lap {row['lap']}: "
            f"{'lane ' + format(lane, '.1f') + ' s, ' if lane is not None else ''}"
            f"stationary {row['stationary_time_s']:.1f} s, "
            f"fuel +{row['fuel_added_l']:.1f} L, "
            f"{'tires changed' if row['tires_changed'] else 'no tires'}")

    logger.on_lap = lap_line
    logger.on_pit = pit_line
    logger.on_session = lambda sid, info: say(
        f"session {sid}: {info.session_type} · {info.car} · {info.track}"
        + (f" · tank {info.tank_l:.1f} L" if info.tank_l else ""))

    period = 1.0 / hz
    waiting_said = False
    last_input_poll = 0.0
    try:
        while not (stop and stop()):
            started = time.perf_counter()
            if started - last_input_poll > 0.25:
                last_input_poll = started
                commands = store.take_commands()
                if poll_inputs:
                    commands += poll_inputs()
                for command, arg in commands:
                    _apply_command(logger, command, arg, say)
            sample = source.read()
            if sample is None:
                if not waiting_said:
                    say("waiting for iRacing ...")
                    waiting_said = True
                logger.finalize()
                time.sleep(1.0)
                continue
            if waiting_said:
                say("connected")
                waiting_said = False
            logger.process(sample)
            spare = period - (time.perf_counter() - started)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        logger.finalize()
        say("logger stopped")


def _apply_command(logger: LapLogger, command: str, arg: dict[str, Any],
                   say: Callable[[str], None]) -> None:
    if command == "tag":
        tag = arg.get("tag", "")
        target = arg.get("target")
        if target is None and tag == logger.tag:
            target = logger.target          # a bare key press keeps the stated target
        try:
            logger.set_tag(tag, target)
        except ValueError as exc:
            say(str(exc))
            return
        say(f"stint tag: {mode_name(logger.tag, logger.target)}")
    elif command == "traffic":
        lap = logger.mark_traffic(last=bool(arg.get("last")))
        say(f"lap {lap} tagged as traffic" if lap is not None else "no lap to tag as traffic")
    else:
        say(f"unknown command {command!r}")
