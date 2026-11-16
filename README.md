# iRacing strategist

A Windows tool for solo iRacing races of about 45 minutes to 3 hours. It answers one
question: is it faster to save fuel or tires and drop a stop, or to push and take it?

This repository holds milestones M1 to M3 of the beta plan:

| Milestone | What it is | Command |
| --- | --- | --- |
| M1 | Stint logger: one row per lap and one record per pit stop, in one SQLite file | `strategist log` |
| M2 | Stint screen (what to drive next) and the fitter | `strategist stint`, `strategist fit` |
| M3 | Solver and plan view, with adjust and re-score | `strategist plan` |

M4 (back-test) and M5 (live re-plan) are not built. The solver already takes the current
lap, fuel and tire age as input, which is what M5 will feed it.

All times the planner prints are **clear-air race time**. It does not see traffic or rivals.

## Install

Python 3.11 or newer, on the sim PC:

```bash
pip install -e .
```

That installs `pyirsdk` on Windows. Try the tool without iRacing first:

```bash
strategist demo
```

```bash
strategist plan --laps 66
```

## Workflow

1. **Check the telemetry names once**, with iRacing in a test session:

   ```bash
   strategist check-vars
   ```

2. **Log.** Leave this running in every session, practice and race:

   ```bash
   strategist log
   ```

   Set the stint tag before each stint. Three ways:
   - type in the logger window: `p` (push), `f 2.6` (fuel-save at 2.6 L per lap), `t`
     (tire-save), `x` (this lap had traffic), `xl` (the last lap had traffic), then Enter;
   - from another terminal: `strategist tag fuel-save 2.6`;
   - a wheel button mapped to a key, with
     `strategist log --hotkeys push=F13,fuel-save=F14,tire-save=F15,traffic=F16`.

   Laps stay `untagged` until you set a tag, and untagged laps are never used for fitting.

3. **Ask what to drive next:**

   ```bash
   strategist stint --minutes 60
   ```

   The default protocol is one push stint, one fuel-save stint at a stated target, and one
   timed pit stop, ten clean laps each. A third stint is asked for only when the ranking of
   the plans turns on something that was not measured.

4. **Plan the race:**

   ```bash
   strategist plan --minutes 60
   ```

   Each stop prints as one line, such as `Pit lap 18, add 42 L, no tires`. Each plan shows
   its gap to the best and the uncertainty of that gap.

5. **Adjust a plan** and have it re-scored: lock a pit lap, set a fuel amount, switch a pace
   mode or a tire change.

   ```bash
   strategist plan --minutes 60 --adjust 1 --pit 1=20 --fuel 1=45 --mode 2=push
   ```

   `strategist plan --minutes 60 -i` keeps a prompt open for repeated adjustments.

Other commands: `sessions`, `laps ID`, `verify ID`, `compare ID ID`, `traffic`.
`strategist COMMAND --help` lists the options. The lap store is `strategist.db` in the
current folder; `--db FILE` or `STRATEGIST_DB` points it elsewhere.

## Race settings

Flags, or a TOML file passed with `--race` (see `race.example.toml`):

| Setting | Flag | Default |
| --- | --- | --- |
| Length | `--laps N` or `--minutes M` | none |
| Tank | `--tank LITRES` | from the logged session |
| Mandatory stops | `--mandatory-stops N` | 0 |
| Tire sets, starting set included | `--tire-sets N` | unlimited |
| No tire changes at all | `--no-tire-changes` | allowed |
| Fuel and tires together or one after the other | `--service together\|sequential` | together |
| Fuel reserve at every stop and at the flag | `--reserve-laps X` | 0.5 lap |
| Timed races: laps of fuel beyond the predicted count | `--timed-extra-laps X` | 1 lap |
| Timed races: one more lap after the lap that ends the clock | `--white-flag-lap` | off |
| Fuel burned before the green flag | `--formation-fuel LITRES` | 0 |

## How it decides

**Logger.** A lap counts as clean unless it left or entered the pits, did not start at the
line, gained incident points, ran under a yellow or black flag, was the first lap of a race,
or had its tag changed part-way. Lap time is iRacing's own value, with the logger's timer as
the fallback.

**Fitter.** The lap-time model is `base + cost of the pace mode + tire wear so far +
fuel weight x fuel on board`. Out-laps, in-laps, incident laps and laps tagged as traffic are
dropped. Every parameter comes with its spread and the number of laps behind it, and a
parameter the stints cannot support is declined:

- a pace mode needs 8 clean laps before it is offered, so a save mode exists only at a level
  you drove;
- the fuel-weight effect is fitted only when stints start from fuel loads at least 20% of
  the tank apart. With one fuel load the fitter reports the net slope of a stint (tire wear
  minus fuel burn-off) and says so;
- the tire-change time is reported as an upper bound when the tires finished inside the
  fuelling time.

**Solver.** A brute-force search over stop count, pit laps, tire changes and pace mode per
stint. Fuel at every stop is just enough for the next stint plus the reserve. In a timed race
the lap count is computed per plan, and plans that run a different number of laps are ranked
by the time each needs for the laps they all complete.

The list holds up to five plans and is never padded. A plan is listed when it could still be
the best (its gap is inside three times its uncertainty), or when it is the best plan with
one stop fewer or one more and costs less than a pit stop. A gap smaller than its uncertainty
is labelled too close to call.

With one fuel load measured, the solver assumes the smallest fuel-weight effect that explains
the stint slope and tells you how large the real effect would have to be for a plan to
overtake plan 1. When that is plausible, no winner is declared and the stint screen asks for
a push stint at a second fuel load.

## Checks to run on your own PC

Everything here was built and tested against a simulated car, not against iRacing. The plan's
"done when" checks need your driving:

- [ ] `strategist check-vars` reports all required variables present.
- [ ] One full practice stint and one full race: `strategist verify ID` reports no missing
      laps and fuel that adds up.
- [ ] One logged pit stop matches a stopwatch within about a second (`strategist laps ID`).
- [ ] Two sessions at the same car and track agree: `strategist compare ID ID`.
- [ ] The fitter declines the fuel-weight effect after a single fuel load: `strategist fit`.
- [ ] The plan for a race you know well matches your own hand calculation.

Known limits:

- **Not run against iRacing.** The logger's handling of the live telemetry (lap counter at
  pit exit and after a reset, the delay before iRacing publishes a lap time, tire-change
  detection) follows the SDK documentation and has only been exercised with the simulator.
- **Global hotkeys** use the Windows hotkey API and have not been run on Windows.
- **Timed-race finish rule.** The default treats the lap in progress when the clock reaches
  zero as the last lap, and fuels for one lap more. If iRacing runs one further lap after
  that, set `--white-flag-lap`. This rule could not be confirmed while building.
- **Thresholds are starting values**: 8 laps per pace mode, 20% of the tank between fuel
  loads, 0.06 s per litre as the largest plausible fuel-weight effect. They are constants at
  the top of `fitter.py` and `solver.py`.

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

| Module | Role |
| --- | --- |
| `telemetry.py` | Telemetry sample, variable list, pyirsdk source |
| `logger.py` | Lap and pit-stop state machine, session checks |
| `inputs.py` | Typed tags and global hotkeys |
| `store.py` | SQLite lap store |
| `sim.py` | Car simulator that stands in for iRacing in tests and `demo` |
| `fitter.py` | Parameters with spread and lap counts |
| `protocol.py` | Stint screen |
| `solver.py` | Pure planner: race, rules, parameters and state in, ranked plans out |
| `planview.py` | Text plan view |
| `cli.py` | Commands |
