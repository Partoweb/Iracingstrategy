"""A small car-and-track simulator that produces telemetry samples.

It stands in for the iRacing client in tests and in `strategist demo`. The
"truth" it drives with is known, so the logger, the fitter and the solver can
all be checked against it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, Union

from .telemetry import Sample, SessionInfo


@dataclass(frozen=True)
class SimMode:
    fuel_per_lap: float     # litres
    cost: float             # seconds per lap slower than push
    deg: float              # seconds of lap time lost per lap of tire age


def _default_modes() -> dict[str, SimMode]:
    return {
        "push": SimMode(fuel_per_lap=3.00, cost=0.0, deg=0.05),
        "fuel-save": SimMode(fuel_per_lap=2.70, cost=0.60, deg=0.05),
        "tire-save": SimMode(fuel_per_lap=2.92, cost=0.70, deg=0.015),
    }


@dataclass(frozen=True)
class SimTruth:
    car: str = "Sim GT3"
    track: str = "Sim Ring"
    tank_l: float = 100.0
    base: float = 100.0             # push lap, new tires, empty tank
    fuel_weight: float = 0.03       # seconds per litre on board
    modes: dict[str, SimMode] = field(default_factory=_default_modes)
    noise_sd: float = 0.15          # lap-to-lap scatter, seconds
    fuel_noise_sd: float = 0.02     # litres per lap
    pit_entry_pct: float = 0.94
    pit_exit_pct: float = 0.05
    stall_pct: float = 0.985
    lane_drive_s: float = 38.0      # pit entry to pit exit without stopping
    refuel_rate: float = 2.6        # litres per second
    service_overhead: float = 1.5   # stationary seconds before fuel flows
    tire_time: float = 20.0
    service_together: bool = True
    out_lap_penalty: float = 1.0    # extra seconds on the lap that leaves the pits

    @property
    def lane_pct(self) -> float:
        return (1.0 - self.pit_entry_pct) + self.pit_exit_pct

    def stationary_time(self, add_l: float, tires: bool) -> float:
        fuel_time = add_l / self.refuel_rate if add_l > 0 else 0.0
        fuel_part = self.service_overhead + fuel_time
        tire_part = self.tire_time if tires else 0.0
        return max(fuel_part, tire_part) if self.service_together else fuel_part + tire_part


# -- script blocks -------------------------------------------------------------

@dataclass(frozen=True)
class Reset:
    """Car appears in its pit stall on new tires with this much fuel."""
    fuel_l: float


@dataclass(frozen=True)
class Grid:
    """Race start: car sits on the grid just before the line."""
    fuel_l: float


@dataclass(frozen=True)
class Run:
    """Drive `laps` full laps. The last one is the in-lap if a Stop follows."""
    mode: str
    laps: int
    tag: str | None = None                  # defaults to the mode name
    target: float | None = None
    incident_laps: tuple[int, ...] = ()     # 1-based within this run
    slow_laps: tuple[int, ...] = ()         # laps held up by traffic


@dataclass(frozen=True)
class Stop:
    add_l: float
    tires: bool = True


@dataclass(frozen=True)
class TagEvent:
    tag: str
    target: float | None = None


Block = Union[Reset, Grid, Run, Stop]
Event = Union[Sample, TagEvent]

SPEED_TRACK = 50.0
SPEED_LANE = 16.0
SLOW_LAP_COST = 1.8
INCIDENT_COST = 2.5


class Simulator:
    """Turns a script of blocks into a stream of samples and tag events."""

    def __init__(self, truth: SimTruth, script: list[Block], *, session_type: str = "Practice",
                 race_laps: int | None = None, race_seconds: float | None = None,
                 hz: float = 20.0, seed: int = 1, session_num: int = 0,
                 lap_time_lag_ticks: int = 2):
        self.truth = truth
        self.script = script
        self.info = SessionInfo(car=truth.car, track=truth.track, session_type=session_type,
                                sub_session_id=None, tank_l=truth.tank_l,
                                race_laps=race_laps, race_seconds=race_seconds)
        self.dt = 1.0 / hz
        self.rng = random.Random(seed)
        self.session_num = session_num
        self.lag = lap_time_lag_ticks

        self.t = 0.0                    # time of the last emitted sample
        self.clock = 0.0                # exact time of the last simulated event
        self.pos = 0.0
        self.lap = 0
        self.fuel = 0.0
        self.wear = 0.0                 # seconds of lap time lost to tire age
        self.incidents = 0
        self.tires_used = 0
        self.pit_sv_flags = 0
        self._last_cross: float | None = None
        self._llt = -1.0
        self._llt_queue: list[tuple[int, float]] = []   # (ticks left, lap time)
        self._resets = 0

    # -- helpers -------------------------------------------------------------

    def _sample(self, *, on_track: bool, on_pit: bool, in_stall: bool, speed: float) -> Sample:
        still_waiting = []
        for ticks, value in self._llt_queue:
            if ticks <= 0:
                self._llt = value
            else:
                still_waiting.append((ticks - 1, value))
        self._llt_queue = still_waiting
        return Sample(
            session_time=self.t, session_num=self.session_num, lap=self.lap,
            lap_last_time=self._llt, fuel_l=self.fuel, on_pit_road=on_pit,
            in_pit_stall=in_stall, is_on_track=on_track, speed=speed,
            lap_dist_pct=self.pos % 1.0, flags=0x0004, incidents=self.incidents,
            track_temp=30.0, air_temp=22.0, wet=False,
            pit_sv_flags=self.pit_sv_flags, tires_used=self.tires_used)

    def _idle(self, seconds: float, *, on_track: bool, on_pit: bool, in_stall: bool
              ) -> Iterator[Sample]:
        end = self.clock + seconds
        while self.t + self.dt <= end:
            self.t += self.dt
            yield self._sample(on_track=on_track, on_pit=on_pit, in_stall=in_stall, speed=0.0)
        self.clock = end

    def _advance(self, target_pos: float, seconds: float, burn: float, *, on_pit: bool,
                 speed: float) -> Iterator[Sample]:
        """Move from the current position to target_pos in `seconds`, burning `burn` litres.

        target_pos may be 1.0, which is the start/finish line. `clock` is the
        exact time of the last event; ticks fall on the fixed sample grid, so
        segment boundaries do not drift with the sample rate.
        """
        start_pos, start_t, start_fuel = self.pos, self.clock, self.fuel
        end_t = start_t + seconds
        while self.t + self.dt < end_t:
            self.t += self.dt
            frac = (self.t - start_t) / seconds
            self.pos = start_pos + (target_pos - start_pos) * frac
            self.fuel = start_fuel - burn * frac
            yield self._sample(on_track=True, on_pit=on_pit, in_stall=False, speed=speed)
        self.pos, self.fuel, self.clock = target_pos, start_fuel - burn, end_t
        if target_pos >= 1.0:
            if self._last_cross is not None:
                self._llt_queue.append((self.lag, end_t - self._last_cross))
            self._last_cross = end_t
            self.lap += 1
            self.pos = 0.0

    # -- the drive -----------------------------------------------------------

    def events(self) -> Iterator[Event]:
        truth = self.truth
        in_pit_lane = False          # car is on pit road at the start of the next lap
        tag: tuple[str, float | None] | None = None
        script = self.script

        for index, block in enumerate(script):
            nxt = script[index + 1] if index + 1 < len(script) else None

            if isinstance(block, Reset):
                if self._last_cross is not None:
                    yield from self._roll_on()
                    self.lap += 1                    # the abandoned lap is gone
                self._last_cross = None
                yield from self._idle(2.0, on_track=False, on_pit=False, in_stall=False)
                self.pos, self.fuel, self.wear = truth.stall_pct, block.fuel_l, 0.0
                if self._resets:
                    self.tires_used += 4             # a reset car is on a new set
                self._resets += 1
                yield from self._idle(1.0, on_track=True, on_pit=True, in_stall=True)
                # Roll from the stall to the line: a partial lap.
                share = (1.0 - truth.stall_pct) / truth.lane_pct
                yield from self._advance(1.0, truth.lane_drive_s * share, 0.02,
                                         on_pit=True, speed=SPEED_LANE)
                in_pit_lane = True

            elif isinstance(block, Grid):
                self.pos, self.fuel, self.wear = 0.995, block.fuel_l, 0.0
                yield from self._idle(2.0, on_track=True, on_pit=False, in_stall=False)
                yield from self._advance(1.0, 1.5, 0.02, on_pit=False, speed=SPEED_TRACK)
                in_pit_lane = False

            elif isinstance(block, Run):
                mode = truth.modes[block.mode]
                wanted = (block.tag or block.mode, block.target)
                if wanted != tag:
                    tag = wanted
                    yield TagEvent(*wanted)
                for n in range(1, block.laps + 1):
                    is_in_lap = n == block.laps and isinstance(nxt, Stop)
                    lap_time = (truth.base + mode.cost + self.wear
                                + truth.fuel_weight * self.fuel
                                + self.rng.gauss(0.0, truth.noise_sd))
                    if n in block.slow_laps:
                        lap_time += SLOW_LAP_COST
                    if n in block.incident_laps:
                        lap_time += INCIDENT_COST
                    burn = mode.fuel_per_lap + self.rng.gauss(0.0, truth.fuel_noise_sd)
                    burn = min(burn, self.fuel)
                    yield from self._drive_lap(lap_time, burn, start_in_lane=in_pit_lane,
                                               in_lap=is_in_lap, stop=nxt if is_in_lap else None,
                                               incident=n in block.incident_laps)
                    self.wear += mode.deg
                    in_pit_lane = is_in_lap

            elif isinstance(block, Stop):
                continue        # handled inside the in-lap of the previous Run

        if self._last_cross is not None:
            yield from self._roll_on()
        yield from self._idle(1.0, on_track=False, on_pit=False, in_stall=False)

    def _roll_on(self) -> Iterator[Sample]:
        """Keep driving for a few seconds past the line before leaving the car."""
        yield from self._advance(self.pos + 0.04, 4.0, 0.1, on_pit=False, speed=SPEED_TRACK)

    def _drive_lap(self, lap_time: float, burn: float, *, start_in_lane: bool, in_lap: bool,
                   stop: Stop | None, incident: bool) -> Iterator[Sample]:
        truth = self.truth
        lane_rate = truth.lane_drive_s / truth.lane_pct      # seconds per unit of lap
        track_from = 0.0
        if start_in_lane:
            # Out-lap: finish the pit lane, then the cold-tire penalty on track.
            yield from self._advance(truth.pit_exit_pct, lane_rate * truth.pit_exit_pct,
                                     burn * truth.pit_exit_pct, on_pit=True, speed=SPEED_LANE)
            track_from = truth.pit_exit_pct
            self.pit_sv_flags = 0
        track_to = truth.pit_entry_pct if in_lap else 1.0
        seconds = lap_time * (track_to - track_from)
        if start_in_lane:
            seconds += truth.out_lap_penalty
        fuel_share = burn * (track_to - track_from)
        if incident:
            half = (track_from + track_to) / 2
            yield from self._advance(half, seconds / 2, fuel_share / 2,
                                     on_pit=False, speed=SPEED_TRACK)
            self.incidents += 1
            yield from self._advance(track_to, seconds / 2, fuel_share / 2,
                                     on_pit=False, speed=SPEED_TRACK)
        else:
            yield from self._advance(track_to, seconds, fuel_share,
                                     on_pit=False, speed=SPEED_TRACK)
        if not in_lap:
            return

        assert stop is not None
        self.pit_sv_flags = (0x0F if stop.tires else 0) | (0x10 if stop.add_l > 0 else 0)
        to_stall = truth.stall_pct - truth.pit_entry_pct
        yield from self._advance(truth.stall_pct, lane_rate * to_stall, burn * to_stall,
                                 on_pit=True, speed=SPEED_LANE)
        yield from self._service(stop)
        rest = 1.0 - truth.stall_pct
        yield from self._advance(1.0, lane_rate * rest, burn * rest,
                                 on_pit=True, speed=SPEED_LANE)

    def _service(self, stop: Stop) -> Iterator[Sample]:
        truth = self.truth
        total = truth.stationary_time(stop.add_l, stop.tires)
        fuel_time = stop.add_l / truth.refuel_rate if stop.add_l > 0 else 0.0
        fuel_from = truth.service_overhead
        if not truth.service_together and stop.tires:
            fuel_from += truth.tire_time          # tires first, then fuel
        start_t, start_fuel = self.clock, self.fuel
        end_t = start_t + total
        while self.t + self.dt < end_t:
            self.t += self.dt
            elapsed = self.t - start_t
            if fuel_time > 0:
                frac = min(1.0, max(0.0, (elapsed - fuel_from) / fuel_time))
                self.fuel = start_fuel + stop.add_l * frac
            yield self._sample(on_track=True, on_pit=True, in_stall=True, speed=0.0)
        self.fuel, self.clock = start_fuel + stop.add_l, end_t
        if stop.tires:
            self.wear = 0.0
            self.tires_used += 4
        # One more stationary tick so the full fuel level is seen at standstill.
        self.t += self.dt
        yield self._sample(on_track=True, on_pit=True, in_stall=True, speed=0.0)
        self.clock = max(self.clock, self.t)


def feed(logger: "object", sim: Simulator) -> None:
    """Run a whole simulated session through a LapLogger."""
    for event in sim.events():
        if isinstance(event, TagEvent):
            logger.set_tag(event.tag, event.target)      # type: ignore[attr-defined]
        else:
            logger.process(event)                        # type: ignore[attr-defined]
    logger.finalize()                                    # type: ignore[attr-defined]
