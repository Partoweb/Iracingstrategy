"""Telemetry samples and the iRacing SDK source.

The logger only ever sees `Sample` and `SessionInfo`, so it can be driven by
the live sim (through pyirsdk) or by the built-in simulator in `sim.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

# Variables the logger cannot work without.
REQUIRED_VARS = (
    "SessionTime",
    "SessionNum",
    "Lap",
    "LapLastLapTime",
    "FuelLevel",
    "OnPitRoad",
    "IsOnTrack",
    "Speed",
    "LapDistPct",
    "SessionFlags",
    "PlayerCarMyIncidentCount",
)

# Variables that improve a record but have a fallback when missing.
OPTIONAL_VARS = {
    "PlayerCarInPitStall": "stationary time falls back to speed on pit road",
    "TrackTempCrew": "track temperature is left empty",
    "AirTemp": "air temperature is left empty",
    "WeatherDeclaredWet": "wet flag falls back to TrackWetness",
    "TrackWetness": "wet flag falls back to WeatherDeclaredWet",
    "PitSvFlags": "tire changes are detected from the tires-used counters only",
    "LFTiresUsed": "tire changes are detected from the pit service request only",
    "RFTiresUsed": "tire changes are detected from the pit service request only",
    "LRTiresUsed": "tire changes are detected from the pit service request only",
    "RRTiresUsed": "tire changes are detected from the pit service request only",
}

TIRE_CHANGE_BITS = 0x0F          # irsdk PitSvFlags: LF, RF, LR, RR tire change

# irsdk Flags bits that make a lap unusable for fitting.
FLAG_YELLOW = 0x0008
FLAG_RED = 0x0010
FLAG_YELLOW_WAVING = 0x0100
FLAG_ONE_TO_GREEN = 0x0200
FLAG_CAUTION = 0x4000
FLAG_CAUTION_WAVING = 0x8000
FLAG_BLACK = 0x010000
FLAG_REPAIR = 0x100000
UNCLEAN_FLAGS = (FLAG_YELLOW | FLAG_RED | FLAG_YELLOW_WAVING | FLAG_ONE_TO_GREEN
                 | FLAG_CAUTION | FLAG_CAUTION_WAVING | FLAG_BLACK | FLAG_REPAIR)


@dataclass(slots=True)
class Sample:
    """One telemetry tick, reduced to what the logger uses."""

    session_time: float
    session_num: int
    lap: int                    # lap the car is currently on (laps started)
    lap_last_time: float        # iRacing's time for the last completed lap, <=0 if none
    fuel_l: float
    on_pit_road: bool
    in_pit_stall: bool | None   # None when the SDK does not provide it
    is_on_track: bool
    speed: float                # m/s
    lap_dist_pct: float
    flags: int
    incidents: int
    track_temp: float | None = None
    air_temp: float | None = None
    wet: bool = False
    pit_sv_flags: int = 0
    tires_used: int | None = None   # sum of the four tires-used counters


@dataclass(slots=True)
class SessionInfo:
    car: str
    track: str
    session_type: str
    sub_session_id: int | None = None
    tank_l: float | None = None
    race_laps: int | None = None
    race_seconds: float | None = None


def check_variables(available: Iterable[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Compare the names this tool reads against the SDK's own variable list.

    Returns (missing required names, [(missing optional name, fallback)]).
    """
    have = set(available)
    missing_required = [v for v in REQUIRED_VARS if v not in have]
    missing_optional = [(v, why) for v, why in OPTIONAL_VARS.items() if v not in have]
    return missing_required, missing_optional


def _to_float(text: Any) -> float | None:
    """Parse session-info values such as '3600.0000 sec' or 'unlimited'."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(text))
    return float(match.group(1)) if match else None


def parse_session_info(weekend: dict[str, Any] | None, driver: dict[str, Any] | None,
                       sessions: dict[str, Any] | None, session_num: int) -> SessionInfo:
    """Build a SessionInfo from the three session-info sections the SDK exposes."""
    weekend = weekend or {}
    driver = driver or {}
    track = str(weekend.get("TrackDisplayName") or weekend.get("TrackName") or "unknown track")
    config = weekend.get("TrackConfigName")
    if config and str(config).strip().lower() not in ("", "null", "none"):
        track = f"{track} - {config}"

    car = "unknown car"
    car_idx = driver.get("DriverCarIdx")
    for entry in driver.get("Drivers") or []:
        if entry.get("CarIdx") == car_idx:
            car = str(entry.get("CarScreenName") or entry.get("CarPath") or car)
            break

    tank = _to_float(driver.get("DriverCarFuelMaxLtr"))
    pct = _to_float(driver.get("DriverCarMaxFuelPct"))
    if tank is not None and pct is not None and 0 < pct <= 1:
        tank *= pct

    session_type = "unknown"
    race_laps: int | None = None
    race_seconds: float | None = None
    for entry in (sessions or {}).get("Sessions") or []:
        if entry.get("SessionNum") == session_num:
            session_type = str(entry.get("SessionType") or session_type)
            laps = _to_float(entry.get("SessionLaps"))
            seconds = _to_float(entry.get("SessionTime"))
            if "race" in session_type.lower():
                # iRacing gives both for some races; a real lap limit is small.
                if laps is not None and laps < 10000:
                    race_laps = int(laps)
                if seconds is not None and seconds < 604800:
                    race_seconds = seconds
            break

    sub = weekend.get("SubSessionID")
    return SessionInfo(car=car, track=track, session_type=session_type,
                       sub_session_id=int(sub) if isinstance(sub, int) else None,
                       tank_l=tank, race_laps=race_laps, race_seconds=race_seconds)


class IRSDKSource:
    """Live telemetry from the iRacing client through pyirsdk (Windows only)."""

    name = "irsdk"

    def __init__(self) -> None:
        try:
            import irsdk  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the platform
            raise RuntimeError(
                "pyirsdk is not installed. Run: pip install pyirsdk") from exc
        self._ir = irsdk.IRSDK()
        self._have: set[str] = set()
        self._connected = False

    def connect(self) -> bool:
        """Try to attach to a running sim. Returns True when data is flowing."""
        ir = self._ir
        if self._connected and not (ir.is_initialized and ir.is_connected):
            ir.shutdown()
            self._connected = False
        if not self._connected:
            try:
                started = ir.startup()
            except Exception:
                # startup() needs the Windows shared-memory API; anywhere else
                # there is no sim to attach to.
                started = False
            if started and ir.is_initialized and ir.is_connected:
                self._connected = True
                self._have = set(ir.var_headers_names)
        return self._connected

    def available_vars(self) -> list[str]:
        return sorted(self._have)

    def close(self) -> None:
        self._ir.shutdown()
        self._connected = False

    def _get(self, name: str, default: Any = None) -> Any:
        return self._ir[name] if name in self._have else default

    def read(self) -> Sample | None:
        """Read one consistent tick, or None when the sim is not sending data."""
        if not self.connect():
            return None
        ir = self._ir
        ir.freeze_var_buffer_latest()
        try:
            session_num = ir["SessionNum"]
            if session_num is None:
                return None
            wet = bool(self._get("WeatherDeclaredWet", False))
            if "WeatherDeclaredWet" not in self._have:
                wetness = self._get("TrackWetness")
                wet = wetness is not None and wetness >= 3   # very_lightly_wet and up
            tires_used = None
            counters = [self._get(v) for v in
                        ("LFTiresUsed", "RFTiresUsed", "LRTiresUsed", "RRTiresUsed")]
            if all(c is not None for c in counters):
                tires_used = int(sum(counters))
            stall = self._get("PlayerCarInPitStall")
            return Sample(
                session_time=float(ir["SessionTime"]),
                session_num=int(session_num),
                lap=int(ir["Lap"]),
                lap_last_time=float(ir["LapLastLapTime"]),
                fuel_l=float(ir["FuelLevel"]),
                on_pit_road=bool(ir["OnPitRoad"]),
                in_pit_stall=None if stall is None else bool(stall),
                is_on_track=bool(ir["IsOnTrack"]),
                speed=float(ir["Speed"]),
                lap_dist_pct=float(ir["LapDistPct"]),
                flags=int(ir["SessionFlags"]) & 0xFFFFFFFF,
                incidents=int(ir["PlayerCarMyIncidentCount"]),
                track_temp=self._get("TrackTempCrew"),
                air_temp=self._get("AirTemp"),
                wet=wet,
                pit_sv_flags=int(self._get("PitSvFlags", 0) or 0),
                tires_used=tires_used,
            )
        finally:
            ir.unfreeze_var_buffer_latest()

    def session_info(self, session_num: int) -> SessionInfo:
        ir = self._ir
        return parse_session_info(ir["WeekendInfo"], ir["DriverInfo"], ir["SessionInfo"],
                                  session_num)
