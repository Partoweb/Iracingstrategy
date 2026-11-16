"""Command line for the strategist: log, tag, fit, stint screen and plan view."""

from __future__ import annotations

import argparse
import sys
import tomllib
from typing import Any

from . import __version__
from .fitter import Fit, compare_fits, fit_parameters
from .logger import TAGS, LapLogger, mode_name, run_logger, verify_session
from .planview import render, render_adjusted
from .protocol import render as render_protocol
from .protocol import stint_progress, stint_protocol
from .solver import Race, Result, rescore, solve
from .store import Store
from .telemetry import OPTIONAL_VARS, REQUIRED_VARS, check_variables


class UserError(Exception):
    """A problem the user can fix; printed without a traceback."""


# -- shared helpers ------------------------------------------------------------------

def _combo(store: Store, args: argparse.Namespace) -> tuple[str, str]:
    car, track = getattr(args, "car", None), getattr(args, "track", None)
    if car and track:
        return car, track
    latest = store.latest_combo()
    if latest is None:
        raise UserError("Nothing logged yet. Run 'strategist log' with iRacing open, "
                        "or 'strategist demo' to try the tool on simulated data.")
    return car or latest[0], track or latest[1]


def _session_ids(store: Store, args: argparse.Namespace, car: str, track: str) -> list[int]:
    ids = store.session_ids(car, track)
    wanted = getattr(args, "sessions", None)
    if wanted:
        missing = sorted(set(wanted) - set(ids))
        if missing:
            raise UserError(f"session(s) {missing} are not logged for {car} at {track}")
        ids = [i for i in ids if i in wanted]
    if not ids:
        raise UserError(f"no sessions logged for {car} at {track}")
    return ids


def _fit(store: Store, args: argparse.Namespace, *, together: bool,
         tank_l: float | None = None) -> Fit:
    car, track = _combo(store, args)
    ids = _session_ids(store, args, car, track)
    return fit_parameters(store.laps_df(ids), store.pits_df(ids), car=car, track=track,
                          session_ids=ids, tank_l=tank_l or store.tank_l(car, track),
                          service_together=together)


RACE_KEYS = {"laps", "minutes", "tank_l", "mandatory_stops", "tire_sets", "tire_changes",
             "service", "reserve_laps", "timed_extra_laps", "white_flag_lap",
             "formation_fuel_l", "max_stops", "car", "track"}


def _race_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the race file and the command-line flags; flags win."""
    settings: dict[str, Any] = {}
    if getattr(args, "race", None):
        try:
            with open(args.race, "rb") as handle:
                settings = tomllib.load(handle)
        except OSError as exc:
            raise UserError(f"cannot read {args.race}: {exc.strerror}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise UserError(f"{args.race} is not valid TOML: {exc}") from exc
        unknown = sorted(set(settings) - RACE_KEYS)
        if unknown:
            raise UserError(f"{args.race}: unknown setting(s) {', '.join(unknown)}")
    for key in ("laps", "minutes", "mandatory_stops", "tire_sets", "service", "reserve_laps",
                "timed_extra_laps", "formation_fuel_l", "max_stops"):
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
            if key in ("laps", "minutes"):
                settings.pop("minutes" if key == "laps" else "laps", None)
    if getattr(args, "tank", None) is not None:
        settings["tank_l"] = args.tank
    if getattr(args, "no_tire_changes", False):
        settings["tire_changes"] = False
    if getattr(args, "white_flag_lap", False):
        settings["white_flag_lap"] = True
    for key in ("car", "track"):
        if settings.get(key) and not getattr(args, key, None):
            setattr(args, key, settings[key])
    return settings


def _has_race(settings: dict[str, Any]) -> bool:
    return "laps" in settings or "minutes" in settings


def _build_race(settings: dict[str, Any], tank_l: float | None) -> Race:
    tank = settings.get("tank_l", tank_l)
    if tank is None:
        raise UserError("Tank size is unknown for this car: give it with --tank LITRES.")
    service = str(settings.get("service", "together")).lower()
    if service not in ("together", "sequential"):
        raise UserError("service must be 'together' or 'sequential'")
    tire_sets = settings.get("tire_sets")
    try:
        return Race(
            tank_l=float(tank), laps=settings.get("laps"), minutes=settings.get("minutes"),
            mandatory_stops=int(settings.get("mandatory_stops", 0)),
            tire_sets=int(tire_sets) if tire_sets else None,
            tire_change_allowed=bool(settings.get("tire_changes", True)),
            service_together=service == "together",
            reserve_laps=float(settings.get("reserve_laps", 0.5)),
            timed_extra_laps=float(settings.get("timed_extra_laps", 1.0)),
            white_flag_lap=bool(settings.get("white_flag_lap", False)),
            formation_fuel_l=float(settings.get("formation_fuel_l", 0.0)),
            max_stops=int(settings.get("max_stops", 4)))
    except ValueError as exc:
        raise UserError(str(exc)) from exc


def _pairs(items: list[str] | None, kind: str, convert) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for item in items or []:
        index, sep, value = item.partition("=")
        try:
            if not sep:
                raise ValueError
            out[int(index)] = convert(value)
        except ValueError:
            raise UserError(f"--{kind} takes NUMBER=VALUE, got {item!r}") from None
    return out


def _yes_no(text: str) -> bool:
    if text.lower() in ("yes", "y", "true", "1", "tires"):
        return True
    if text.lower() in ("no", "n", "false", "0", "none"):
        return False
    raise ValueError


# -- commands ---------------------------------------------------------------------------

def cmd_log(args: argparse.Namespace) -> int:
    from .inputs import HELP, ConsoleInput, GlobalHotkeys, parse_hotkeys
    from .telemetry import IRSDKSource

    try:
        source = IRSDKSource()
    except RuntimeError as exc:
        raise UserError(str(exc)) from exc
    pollers = []
    if args.hotkeys:
        try:
            pollers.append(GlobalHotkeys(parse_hotkeys(args.hotkeys)))
        except (ValueError, RuntimeError) as exc:
            raise UserError(f"hotkeys: {exc}") from exc
    pollers.append(ConsoleInput())

    def poll() -> list[tuple[str, dict[str, Any]]]:
        return [command for poller in pollers for command in poller.poll()]

    with Store(args.db) as store:
        checked = False

        def on_lap(logger: LapLogger, row: dict[str, Any]) -> None:
            laps = store.laps_df([row["session_id"]])
            line = stint_progress(laps, row["session_id"], row["stint_number"])
            if line:
                print(f"         {line}")

        def read_checked() -> Any:
            nonlocal checked
            sample = source.read()
            if sample is not None and not checked:
                checked = True
                missing, optional = check_variables(source.available_vars())
                if missing:
                    raise UserError("iRacing does not provide: " + ", ".join(missing)
                                    + ". Run 'strategist check-vars'.")
                for name, fallback in optional:
                    print(f"note: {name} is not available; {fallback}")
            return sample

        class Checked:
            name = source.name
            session_info = staticmethod(source.session_info)
            read = staticmethod(read_checked)

        print(f"logging to {store.path}.  {HELP}")
        if args.tag:
            store.send_command("tag", {"tag": args.tag, "target": args.target})
        run_logger(Checked(), store, hz=args.hz, poll_inputs=poll, on_lap=on_lap)
    return 0


def cmd_check_vars(args: argparse.Namespace) -> int:
    from .telemetry import IRSDKSource

    try:
        source = IRSDKSource()
    except RuntimeError as exc:
        raise UserError(str(exc)) from exc
    if not source.connect():
        raise UserError("iRacing is not running (or not in a session). Start a test "
                        "session and run this again.")
    available = source.available_vars()
    missing, optional = check_variables(available)
    print(f"The SDK lists {len(available)} variables.")
    for name in REQUIRED_VARS:
        print(f"  {'MISSING' if name in missing else 'ok     '}  {name}")
    gone = {name for name, _ in optional}
    for name, fallback in OPTIONAL_VARS.items():
        if name in gone:
            print(f"  missing  {name}  (optional: {fallback})")
        else:
            print(f"  ok       {name}  (optional)")
    source.close()
    if missing:
        print("The logger cannot run without the variables marked MISSING.")
        return 1
    print("All required variables are present.")
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        store.send_command("tag", {"tag": args.tag, "target": args.target})
    print(f"sent: {mode_name(args.tag, args.target)} (a running logger picks it up within "
          f"a second)")
    return 0


def cmd_traffic(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        if args.session is not None or args.lap is not None:
            if args.session is None or args.lap is None:
                raise UserError("give both --session and --lap, or neither")
            if not store.mark_traffic(args.session, args.lap, not args.clear):
                raise UserError(f"session {args.session} has no lap {args.lap}")
            print(f"session {args.session} lap {args.lap}: "
                  f"{'traffic tag removed' if args.clear else 'tagged as traffic'}")
        else:
            store.send_command("traffic", {"last": args.last})
            print("sent: traffic tag for the " + ("last" if args.last else "current") + " lap")
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        rows = store.sessions()
    if not rows:
        print("No sessions logged yet.")
        return 0
    print(f"{'id':>3}  {'started (UTC)':19}  {'type':12}  {'laps':>4}  {'stops':>5}  car · track")
    for r in rows:
        print(f"{r['id']:>3}  {r['started_at'][:19]:19}  {r['session_type'][:12]:12}  "
              f"{r['n_laps']:>4}  {r['n_stops']:>5}  {r['car']} · {r['track']}")
    return 0


def cmd_laps(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        if store.session(args.session) is None:
            raise UserError(f"there is no session {args.session}")
        laps = store.laps_df([args.session])
        pits = store.pits_df([args.session])
    print(f"{'lap':>3}  {'time':>9}  {'fuel':>5}  {'left':>5}  {'stint':>5}  {'tires':>5}  "
          f"{'tag':14}  state")
    for row in laps.itertuples():
        time = "    --   " if row.lap_time_s != row.lap_time_s else f"{row.lap_time_s:9.3f}"
        state = "clean" if row.clean else row.unclean_reason
        if row.traffic:
            state += " +traffic"
        target = None if row.tag_target != row.tag_target else row.tag_target
        print(f"{row.lap_number:>3}  {time}  {row.fuel_used_l:5.2f}  {row.fuel_at_line_l:5.1f}  "
              f"{row.stint_number:>2}/{row.lap_in_stint:<2}  {row.tire_age:>5}  "
              f"{mode_name(row.stint_tag, target):14}  {state}")
    for stop in pits.itertuples():
        lane = "never left the pits" if stop.pit_lane_time_s != stop.pit_lane_time_s \
            else f"lane {stop.pit_lane_time_s:.1f} s"
        print(f"pit stop on lap {stop.lap}: {lane}, stationary {stop.stationary_time_s:.1f} s, "
              f"fuel +{stop.fuel_added_l:.1f} L in {stop.fuel_time_s:.1f} s, "
              f"{'tires changed' if stop.tires_changed else 'no tires'}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        if store.session(args.session) is None:
            raise UserError(f"there is no session {args.session}")
        problems = verify_session(store, args.session)
        laps = store.laps_df([args.session])
    if problems:
        print(f"session {args.session}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"session {args.session}: {len(laps)} laps, none missing, and fuel used per lap "
          f"adds up to the drop in the tank.")
    return 0


def _print_fit(fit: Fit) -> None:
    print(f"{fit.car} · {fit.track} · sessions {', '.join(map(str, fit.session_ids))} · "
          f"{fit.n_clean_laps} clean tagged laps")
    print(f"{'parameter':18} {'mode':15} {'value':>10} {'spread':>9} {'n':>4}  note")
    for p in fit.params:
        if p.status != "ok":
            print(f"{p.name:18} {p.mode or '':15} {'declined':>10} {'':>9} {p.n:>4}  {p.note}")
            continue
        spread = "" if p.spread is None else f"± {p.spread:.4f}"
        print(f"{p.name:18} {p.mode or '':15} {p.value:>10.4f} {spread:>9} {p.n:>4}  "
              f"{p.unit}{'; ' + p.note if p.note else ''}")
    for note in fit.notes:
        print(f"note: {note}")


def cmd_fit(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        fit = _fit(store, args, together=args.service != "sequential")
        fit_id = store.save_fit(fit)
    _print_fit(fit)
    print(f"saved as fit {fit_id}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        fits = []
        for session_id in (args.a, args.b):
            row = store.session(session_id)
            if row is None:
                raise UserError(f"there is no session {session_id}")
            fits.append(fit_parameters(
                store.laps_df([session_id]), store.pits_df([session_id]), car=row["car"],
                track=row["track"], session_ids=[session_id], tank_l=row["tank_l"],
                service_together=args.service != "sequential"))
    a, b = fits
    if (a.car, a.track) != (b.car, b.track):
        raise UserError("the two sessions are not the same car and track")
    rows = compare_fits(a, b)
    if not rows:
        print("No parameter could be fitted in both sessions.")
        return 1
    print(f"{'parameter':18} {'mode':15} {'session ' + str(args.a):>12} "
          f"{'session ' + str(args.b):>12} {'diff':>9} {'2x spread':>10}  verdict")
    disagree = 0
    for r in rows:
        if r.agree is None:
            verdict, bound = "spread unknown", ""
        else:
            verdict = "agree" if r.agree else "DISAGREE"
            bound = f"{2 * r.spread:10.4f}"
            disagree += not r.agree
        print(f"{r.name:18} {r.mode or '':15} {r.a:>12.4f} {r.b:>12.4f} {r.b - r.a:>+9.4f} "
              f"{bound:>10}  {verdict}")
    for fit, session_id in ((a, args.a), (b, args.b)):
        weight = fit.param("fuel_weight")
        if weight is not None and weight.status == "declined":
            print(f"session {session_id}: fuel weight declined ({weight.note})")
    print("All parameters agree within their spread." if not disagree
          else f"{disagree} parameter(s) disagree by more than twice their combined spread.")
    return 1 if disagree else 0


def cmd_stint(args: argparse.Namespace) -> int:
    settings = _race_settings(args)
    together = str(settings.get("service", "together")).lower() != "sequential"
    with Store(args.db) as store:
        car, track = _combo(store, args)
        ids = _session_ids(store, args, car, track)
        tank = settings.get("tank_l", store.tank_l(car, track))
        fit = _fit(store, args, together=together, tank_l=tank)
        race = result = None
        if _has_race(settings):
            race = _build_race(settings, tank)
            result = solve(race, fit)
        proto = stint_protocol(store.laps_df(ids), fit, tank_l=tank, race=race, result=result)
    print(f"{car} · {track}" + (f" · {race.minutes:g} min race" if race and race.timed
                                 else f" · {race.laps}-lap race" if race else ""))
    print(render_protocol(proto))
    return 0


def _adjust(result: Result, race: Race, fit: Fit, number: int, pit: dict[int, int],
            fuel: dict[int, float], mode: dict[int, str], tires: dict[int, bool]) -> str:
    if not 1 <= number <= len(result.plans):
        raise UserError(f"there is no plan {number}; choose 1 to {len(result.plans)}")
    adjusted = rescore(race, fit, result.plans[number - 1], pit_laps=pit, fuel=fuel,
                       modes=mode, tires=tires, reference=result.plans[0])
    return render_adjusted(adjusted, number)


def _interactive(result: Result, race: Race, fit: Fit) -> None:
    print("\nAdjust a plan:  PLAN pit STOP=LAP  fuel STOP=LITRES  mode STINT=MODE  "
          "tires STOP=yes|no   (fuel 0=... sets the start fuel; q quits)")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line in ("q", "quit", "exit"):
            return
        if not line:
            continue
        words = line.split()
        try:
            number = int(words[0])
            groups: dict[str, list[str]] = {"pit": [], "fuel": [], "mode": [], "tires": []}
            kind = None
            for word in words[1:]:
                if word in groups:
                    kind = word
                elif kind is None:
                    raise UserError("start with the plan number, then pit, fuel, mode or tires")
                else:
                    groups[kind].append(word)
            print(_adjust(result, race, fit, number, _pairs(groups["pit"], "pit", int),
                          _pairs(groups["fuel"], "fuel", float), _pairs(groups["mode"], "mode", str),
                          _pairs(groups["tires"], "tires", _yes_no)))
        except ValueError:
            print("start with the plan number, for example: 1 pit 1=20")
        except UserError as exc:
            print(exc)


def cmd_plan(args: argparse.Namespace) -> int:
    settings = _race_settings(args)
    if not _has_race(settings):
        raise UserError("Give the race length: --laps N or --minutes M (or a --race file).")
    together = str(settings.get("service", "together")).lower() != "sequential"
    with Store(args.db) as store:
        car, track = _combo(store, args)
        tank = settings.get("tank_l", store.tank_l(car, track))
        fit = _fit(store, args, together=together, tank_l=tank)
    race = _build_race(settings, tank)
    result = solve(race, fit)
    print(render(result, race, fit))
    wants_adjust = args.adjust is not None or args.pit or args.fuel or args.mode or args.tires
    if result.kind != "plans":
        if wants_adjust:
            raise UserError("there is no plan to adjust")
        return 0 if result.kind == "start-fuel" else 1
    if wants_adjust:
        print()
        print(_adjust(result, race, fit, args.adjust or 1, _pairs(args.pit, "pit", int),
                      _pairs(args.fuel, "fuel", float), _pairs(args.mode, "mode", str),
                      _pairs(args.tires, "tires", _yes_no)))
    if args.interactive:
        _interactive(result, race, fit)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .sim import Grid, Reset, Run, Simulator, SimTruth, Stop, feed

    truth = SimTruth()
    practice = [
        Reset(100.0), Run("push", 13),
        Reset(100.0), Run("fuel-save", 13, target=2.7),
        Reset(100.0), Run("push", 6), Stop(add_l=18.0, tires=True), Run("push", 5),
    ]
    if args.two_loads:
        practice += [Reset(55.0), Run("push", 12)]
    race = [Grid(100.0), Run("push", 20), Stop(add_l=60.0, tires=True), Run("push", 16)]
    with Store(args.db) as store:
        for script, kind, kwargs in ((practice, "Practice", {}),
                                     (race, "Race", {"race_laps": 36})):
            sim = Simulator(truth, script, session_type=kind, seed=args.seed, **kwargs)
            logger = LapLogger(store=store, info_provider=lambda num, sim=sim: sim.info,
                               source="sim")
            feed(logger, sim)
        print(f"Wrote a simulated practice session and a 36-lap race for {truth.car} at "
              f"{truth.track} to {store.path}.")
    print("Try:  strategist sessions   ·   strategist fit   ·   strategist stint --laps 66"
          "   ·   strategist plan --laps 66")
    return 0


# -- argument parsing ----------------------------------------------------------------------

def _add_combo(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--car", help="car name as logged (default: the latest session's)")
    parser.add_argument("--track", help="track name as logged (default: the latest session's)")
    parser.add_argument("--sessions", type=int, nargs="+", metavar="ID",
                        help="use only these sessions (default: all for the car and track)")


def _add_race(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--race", metavar="FILE", help="race settings as a TOML file")
    parser.add_argument("--laps", type=int, help="race length in laps")
    parser.add_argument("--minutes", type=float, help="race length in minutes")
    parser.add_argument("--tank", type=float, metavar="LITRES", help="tank size")
    parser.add_argument("--mandatory-stops", type=int, dest="mandatory_stops")
    parser.add_argument("--tire-sets", type=int, dest="tire_sets",
                        help="tire sets available, the starting set included")
    parser.add_argument("--no-tire-changes", action="store_true", dest="no_tire_changes")
    parser.add_argument("--service", choices=("together", "sequential"),
                        help="fuel and tires at the same time, or one after the other")
    parser.add_argument("--reserve-laps", type=float, dest="reserve_laps")
    parser.add_argument("--timed-extra-laps", type=float, dest="timed_extra_laps")
    parser.add_argument("--white-flag-lap", action="store_true", dest="white_flag_lap",
                        help="timed race: one more lap after the lap that ends the clock")
    parser.add_argument("--formation-fuel", type=float, dest="formation_fuel_l",
                        metavar="LITRES", help="fuel burned before the green flag")
    parser.add_argument("--max-stops", type=int, dest="max_stops")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strategist",
        description="Stint logger, fitter and pit-strategy planner for solo iRacing races.")
    parser.add_argument("--version", action="version", version=f"strategist {__version__}")
    parser.add_argument("--db", help="lap store file (default: strategist.db, or "
                                     "$STRATEGIST_DB)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("log", help="log every lap and pit stop from the running sim")
    p.add_argument("--hz", type=float, default=60.0, help="samples per second (default 60)")
    p.add_argument("--tag", choices=TAGS, help="stint tag to start with")
    p.add_argument("--target", type=float, help="stated target for the tag, e.g. L per lap")
    p.add_argument("--hotkeys", metavar="SPEC",
                   help="global hotkeys, e.g. push=F13,fuel-save=F14,tire-save=F15,traffic=F16")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("check-vars", help="check the telemetry names against the SDK's list")
    p.set_defaults(func=cmd_check_vars)

    p = sub.add_parser("tag", help="set the stint tag of a running logger")
    p.add_argument("tag", choices=TAGS)
    p.add_argument("target", type=float, nargs="?", help="stated target, e.g. L per lap")
    p.set_defaults(func=cmd_tag)

    p = sub.add_parser("traffic", help="tag a lap as held up by traffic")
    p.add_argument("--last", action="store_true", help="the last completed lap (live)")
    p.add_argument("--session", type=int)
    p.add_argument("--lap", type=int)
    p.add_argument("--clear", action="store_true", help="remove the tag instead")
    p.set_defaults(func=cmd_traffic)

    p = sub.add_parser("sessions", help="list logged sessions")
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("laps", help="show the laps and pit stops of a session")
    p.add_argument("session", type=int)
    p.set_defaults(func=cmd_laps)

    p = sub.add_parser("verify", help="check a session for missing laps and fuel that "
                                      "does not add up")
    p.add_argument("session", type=int)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("fit", help="fit parameters from the tagged stints")
    _add_combo(p)
    p.add_argument("--service", choices=("together", "sequential"), default="together")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("compare", help="do two sessions give the same parameters?")
    p.add_argument("a", type=int, help="session id")
    p.add_argument("b", type=int, help="session id")
    p.add_argument("--service", choices=("together", "sequential"), default="together")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("stint", help="stint screen: what to drive next, and how many laps")
    _add_combo(p)
    _add_race(p)
    p.set_defaults(func=cmd_stint)

    p = sub.add_parser("plan", help="plan view: ranked strategies for a race")
    _add_combo(p)
    _add_race(p)
    p.add_argument("--adjust", type=int, metavar="PLAN", help="plan number to adjust")
    p.add_argument("--pit", action="append", metavar="STOP=LAP", help="lock a pit lap")
    p.add_argument("--fuel", action="append", metavar="STOP=LITRES",
                   help="set the fuel added at a stop (0=... sets the start fuel)")
    p.add_argument("--mode", action="append", metavar="STINT=MODE",
                   help="switch the pace mode of a stint")
    p.add_argument("--tires", action="append", metavar="STOP=yes|no",
                   help="change tires at a stop or not")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="keep adjusting plans at a prompt")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("demo", help="write simulated sessions to try the tool without iRacing")
    p.add_argument("--two-loads", action="store_true",
                   help="add a push stint at a second fuel load")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
