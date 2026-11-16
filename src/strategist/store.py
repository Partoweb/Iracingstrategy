"""Lap store: one SQLite file holding sessions, laps, pit stops and fits."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_DB = "strategist.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY,
    started_at      TEXT NOT NULL,
    source          TEXT NOT NULL,          -- 'irsdk' or 'sim'
    car             TEXT NOT NULL,
    track           TEXT NOT NULL,
    session_type    TEXT NOT NULL,          -- Practice, Qualify, Race, ...
    sub_session_id  INTEGER,
    session_num     INTEGER,
    tank_l          REAL,                   -- usable tank in litres
    race_laps       INTEGER,                -- NULL when the session is timed
    race_seconds    REAL                    -- NULL when the session is lap-limited
);

CREATE TABLE IF NOT EXISTS laps (
    id              INTEGER PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES sessions(id),
    lap_number      INTEGER NOT NULL,
    lap_time_s      REAL,                   -- NULL when the lap was not completed
    lap_time_source TEXT NOT NULL,          -- 'sim' (iRacing's own time), 'timer' or 'none'
    session_time_s  REAL NOT NULL,          -- session clock when the lap ended
    fuel_start_l    REAL NOT NULL,          -- fuel when the lap started
    fuel_at_line_l  REAL NOT NULL,          -- fuel when the lap ended
    fuel_added_l    REAL NOT NULL,          -- fuel taken on during the lap
    fuel_used_l     REAL NOT NULL,          -- fuel_start - fuel_at_line + fuel_added
    stint_number    INTEGER NOT NULL,
    lap_in_stint    INTEGER NOT NULL,       -- 1 for the first lap of a stint
    tire_age        INTEGER NOT NULL,       -- laps already done on this tire set
    stint_tag       TEXT NOT NULL,          -- push, fuel-save, tire-save, untagged
    tag_target      REAL,                   -- stated target, e.g. litres per lap
    track_temp_c    REAL,
    air_temp_c      REAL,
    wet             INTEGER NOT NULL,
    pit_road        TEXT NOT NULL,          -- none, in, out, in+out
    flags           INTEGER NOT NULL,       -- session flag bits seen during the lap
    incidents       INTEGER NOT NULL,       -- incident points gained during the lap
    clean           INTEGER NOT NULL,
    unclean_reason  TEXT NOT NULL,          -- comma-separated, empty when clean
    traffic         INTEGER NOT NULL DEFAULT 0,
    UNIQUE (session_id, lap_number)
);

CREATE TABLE IF NOT EXISTS pit_stops (
    id                INTEGER PRIMARY KEY,
    session_id        INTEGER NOT NULL REFERENCES sessions(id),
    lap               INTEGER NOT NULL,     -- lap on which the car entered pit road
    entry_time_s      REAL NOT NULL,        -- session clock at pit entry
    pit_lane_time_s   REAL,                 -- pit entry to pit exit; NULL if never exited
    stationary_time_s REAL NOT NULL,
    fuel_time_s       REAL NOT NULL,        -- time during which the fuel level rose
    fuel_delay_s      REAL,                 -- standstill until fuel started to flow
    fuel_added_l      REAL NOT NULL,
    tires_changed     INTEGER NOT NULL,
    entry_pct         REAL,                 -- lap distance fraction at pit entry
    exit_pct          REAL                  -- lap distance fraction at pit exit
);

CREATE TABLE IF NOT EXISTS fits (
    id          INTEGER PRIMARY KEY,
    fitted_at   TEXT NOT NULL,
    car         TEXT NOT NULL,
    track       TEXT NOT NULL,
    session_ids TEXT NOT NULL,              -- JSON list
    fit_json    TEXT NOT NULL               -- the full fit, as the solver reads it
);

CREATE TABLE IF NOT EXISTS parameters (
    id      INTEGER PRIMARY KEY,
    fit_id  INTEGER NOT NULL REFERENCES fits(id),
    name    TEXT NOT NULL,
    mode    TEXT,
    value   REAL,
    spread  REAL,
    n       INTEGER NOT NULL,
    unit    TEXT NOT NULL,
    status  TEXT NOT NULL,                  -- 'ok' or 'declined'
    note    TEXT NOT NULL                   -- why it was declined, or what it refers to
);

CREATE TABLE IF NOT EXISTS control (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,
    command     TEXT NOT NULL,              -- 'tag' or 'traffic'
    arg         TEXT NOT NULL,              -- JSON
    consumed    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS laps_session ON laps(session_id, lap_number);
CREATE INDEX IF NOT EXISTS pits_session ON pit_stops(session_id, lap);
"""

LAP_COLUMNS = (
    "session_id", "lap_number", "lap_time_s", "lap_time_source", "session_time_s",
    "fuel_start_l", "fuel_at_line_l", "fuel_added_l", "fuel_used_l",
    "stint_number", "lap_in_stint", "tire_age", "stint_tag", "tag_target",
    "track_temp_c", "air_temp_c", "wet", "pit_road", "flags", "incidents",
    "clean", "unclean_reason", "traffic",
)

PIT_COLUMNS = (
    "session_id", "lap", "entry_time_s", "pit_lane_time_s", "stationary_time_s",
    "fuel_time_s", "fuel_delay_s", "fuel_added_l", "tires_changed", "entry_pct", "exit_pct",
)


def default_db_path() -> str:
    return os.environ.get("STRATEGIST_DB", DEFAULT_DB)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """Thin wrapper around the SQLite file. Safe to open from two processes."""

    def __init__(self, path: str | Path | None = None):
        self.path = str(path) if path is not None else default_db_path()
        self.db = sqlite3.connect(self.path, timeout=10.0)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- sessions ---------------------------------------------------------

    def add_session(self, *, source: str, car: str, track: str, session_type: str,
                    sub_session_id: int | None = None, session_num: int | None = None,
                    tank_l: float | None = None, race_laps: int | None = None,
                    race_seconds: float | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO sessions (started_at, source, car, track, session_type,"
            " sub_session_id, session_num, tank_l, race_laps, race_seconds)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (_now(), source, car, track, session_type, sub_session_id, session_num,
             tank_l, race_laps, race_seconds))
        self.db.commit()
        return int(cur.lastrowid)

    def session(self, session_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()

    def sessions(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM laps l WHERE l.session_id=s.id) AS n_laps,"
            " (SELECT COUNT(*) FROM pit_stops p WHERE p.session_id=s.id) AS n_stops"
            " FROM sessions s ORDER BY s.id").fetchall()

    def latest_combo(self) -> tuple[str, str] | None:
        row = self.db.execute(
            "SELECT car, track FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        return (row["car"], row["track"]) if row else None

    def session_ids(self, car: str, track: str) -> list[int]:
        rows = self.db.execute(
            "SELECT id FROM sessions WHERE car=? AND track=? ORDER BY id", (car, track))
        return [r["id"] for r in rows]

    def tank_l(self, car: str, track: str) -> float | None:
        row = self.db.execute(
            "SELECT tank_l FROM sessions WHERE car=? AND track=? AND tank_l IS NOT NULL"
            " ORDER BY id DESC LIMIT 1", (car, track)).fetchone()
        return float(row["tank_l"]) if row else None

    # -- laps and pit stops -----------------------------------------------

    def add_lap(self, row: dict[str, Any]) -> None:
        cols = ",".join(LAP_COLUMNS)
        marks = ",".join("?" for _ in LAP_COLUMNS)
        self.db.execute(
            f"INSERT OR REPLACE INTO laps ({cols}) VALUES ({marks})",
            tuple(row.get(c) for c in LAP_COLUMNS))
        self.db.commit()

    def add_pit_stop(self, row: dict[str, Any]) -> None:
        cols = ",".join(PIT_COLUMNS)
        marks = ",".join("?" for _ in PIT_COLUMNS)
        self.db.execute(
            f"INSERT INTO pit_stops ({cols}) VALUES ({marks})",
            tuple(row.get(c) for c in PIT_COLUMNS))
        self.db.commit()

    def mark_traffic(self, session_id: int, lap_number: int, traffic: bool = True) -> bool:
        cur = self.db.execute(
            "UPDATE laps SET traffic=? WHERE session_id=? AND lap_number=?",
            (int(traffic), session_id, lap_number))
        self.db.commit()
        return cur.rowcount > 0

    def laps_df(self, session_ids: Iterable[int]) -> pd.DataFrame:
        ids = list(session_ids)
        if not ids:
            return pd.DataFrame(columns=["id", *LAP_COLUMNS])
        marks = ",".join("?" for _ in ids)
        return pd.read_sql_query(
            f"SELECT * FROM laps WHERE session_id IN ({marks})"
            " ORDER BY session_id, lap_number", self.db, params=ids)

    def pits_df(self, session_ids: Iterable[int]) -> pd.DataFrame:
        ids = list(session_ids)
        if not ids:
            return pd.DataFrame(columns=["id", *PIT_COLUMNS])
        marks = ",".join("?" for _ in ids)
        return pd.read_sql_query(
            f"SELECT * FROM pit_stops WHERE session_id IN ({marks})"
            " ORDER BY session_id, entry_time_s", self.db, params=ids)

    # -- fits ---------------------------------------------------------------

    def save_fit(self, fit: Any) -> int:
        """Store a fitter.Fit and its parameter rows. Returns the fit id."""
        cur = self.db.execute(
            "INSERT INTO fits (fitted_at, car, track, session_ids, fit_json)"
            " VALUES (?,?,?,?,?)",
            (_now(), fit.car, fit.track, json.dumps(fit.session_ids),
             json.dumps(fit.to_dict())))
        fit_id = int(cur.lastrowid)
        self.db.executemany(
            "INSERT INTO parameters (fit_id, name, mode, value, spread, n, unit, status,"
            " note) VALUES (?,?,?,?,?,?,?,?,?)",
            [(fit_id, p.name, p.mode, p.value, p.spread, p.n, p.unit, p.status, p.note)
             for p in fit.params])
        self.db.commit()
        return fit_id

    def latest_fit_json(self, car: str, track: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT fit_json FROM fits WHERE car=? AND track=? ORDER BY id DESC LIMIT 1",
            (car, track)).fetchone()
        return json.loads(row["fit_json"]) if row else None

    # -- control channel (typed commands for a running logger) ------------

    def send_command(self, command: str, arg: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO control (created_at, command, arg) VALUES (?,?,?)",
            (_now(), command, json.dumps(arg)))
        self.db.commit()

    def take_commands(self) -> list[tuple[str, dict[str, Any]]]:
        rows = self.db.execute(
            "SELECT id, command, arg FROM control WHERE consumed=0 ORDER BY id").fetchall()
        if not rows:
            return []
        self.db.executemany("UPDATE control SET consumed=1 WHERE id=?",
                            [(r["id"],) for r in rows])
        self.db.commit()
        return [(r["command"], json.loads(r["arg"])) for r in rows]
