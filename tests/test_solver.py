import itertools
import math
import random

import pytest

from conftest import log_session, make_fit, practice_script, truth_fit
from strategist.fitter import fit_parameters
from strategist.planview import render, render_adjusted
from strategist.sim import Grid, Run, SimTruth, Stop
from strategist.solver import (PlanSpec, Race, State, _evaluate, _lens_from_pit_laps, _Model,
                               _pit_lap_rows, _count_rows, _shape_times, rescore, solve)


def fitted(store, script, seed=2, **kwargs):
    truth = SimTruth()
    sid = log_session(store, truth, script, seed=seed)
    return fit_parameters(store.laps_df([sid]), store.pits_df([sid]), car=truth.car,
                          track=truth.track, tank_l=truth.tank_l, **kwargs)


# -- hand checks -------------------------------------------------------------------

def test_one_stop_race_matches_a_hand_calculation():
    # 30 laps at 100.0 s, 3 L per lap, 60 L tank, no tire wear, no fuel weight.
    # Fuel: 30 laps + 0.5 lap reserve = 91.5 L, so one stop. A stint can run
    # floor(60 / 3 - 0.5) = 19 laps. Refuelling costs 1 s per 2.5 L, so the
    # fastest plan adds as little as it can: stop on lap 19, add 11 laps = 33 L.
    fit = make_fit(base=100.0, modes={"push": 3.0}, pit_fixed=25.0, rate=2.5)
    result = solve(Race(tank_l=60.0, laps=30), fit)
    assert result.kind == "plans"
    best = result.plans[0]
    assert best.spec.lens == (19, 11)
    assert best.start_fuel_l == pytest.approx(3.0 * 19.5)
    assert best.stops[0].line() == "Pit lap 19, add 33 L, no tires"
    assert best.time_s == pytest.approx(30 * 100.0 + 25.0 + 33.0 / 2.5)
    assert best.laps == 30


def test_tire_wear_and_fuel_weight_match_a_hand_calculation():
    # Two stints of 10 laps on new tires, wear 0.1 s per lap of age, fuel weight
    # 0.02 s per litre relative to 50 L, 2 L per lap, reserve 0.5 lap.
    fit = make_fit(base=90.0, slope=0.1, weight=0.02, ref=50.0, modes={"push": 2.0},
                   pit_fixed=20.0, rate=2.0, overhead=1.0, tire_time=15.0)
    model = _Model(fit, Race(tank_l=40.0, laps=20), State())
    plan = _evaluate(model, PlanSpec(lens=(10, 10), modes=("push", "push"), tires=(True,)))
    start = 2.0 * 10.5                              # 21 L on board at the green
    stint = sum(90.0 + 0.1 * j + 0.02 * (start - 2.0 * j - 50.0) for j in range(10))
    add = 2.0 * 10.5 - 1.0                          # second stint's need minus the reserve left
    stop = 20.0 + max(1.0 + add / 2.0, 15.0)
    assert plan.start_fuel_l == pytest.approx(start)
    assert plan.stops[0].add_l == pytest.approx(add)
    assert plan.time_s == pytest.approx(2 * stint + stop)


def test_no_stop_race_prints_only_a_start_fuel_number():
    fit = make_fit(modes={"push": 3.0, "fuel-save@2.7": 2.7}, costs={"fuel-save@2.7": 0.6})
    race = Race(tank_l=100.0, laps=20, formation_fuel_l=2.0)
    result = solve(race, fit)
    assert result.kind == "start-fuel"
    assert result.start_fuel_l == pytest.approx(3.0 * 20.5 + 2.0)
    text = render(result, race, fit)
    assert "No stop needed. Start fuel: 64 L" in text
    assert "Pit lap" not in text and "1." not in text


def test_no_stop_race_needs_no_pit_data():
    fit = make_fit(pit_fixed=None)
    assert solve(Race(tank_l=100.0, laps=20), fit).kind == "start-fuel"
    needs_stop = solve(Race(tank_l=100.0, laps=60), fit)
    assert needs_stop.kind == "no-plan" and "pit stop" in needs_stop.reason


def test_no_push_stint_means_no_plan():
    fit = make_fit()
    fit.columns, fit.theta = [], []
    result = solve(Race(tank_l=100.0, laps=30), fit)
    assert result.kind == "no-plan" and "push stint" in result.reason


# -- search against exhaustive evaluation ---------------------------------------------

@pytest.mark.parametrize("weight,together", [(0.03, True), (None, True), (0.03, False)])
def test_fast_search_agrees_with_lap_by_lap_evaluation(weight, together):
    fit = make_fit(base=100.0, slope=0.05 if weight else -0.04, weight=weight, ref=90.0,
                   modes={"push": 3.0, "fuel-save@2.7": 2.7, "tire-save": 2.9},
                   costs={"fuel-save@2.7": 0.6, "tire-save": 0.7}, tire_save_slope=0.015,
                   pit_fixed=27.0, rate=2.6, overhead=1.5, tire_time=20.0, together=together)
    race = Race(tank_l=100.0, laps=47, service_together=together)
    model = _Model(fit, race, State())
    rng = random.Random(5)
    modes = list(fit.modes)
    for stops in (0, 1, 2, 3):
        caps = [40] * (stops + 1)
        rows = _pit_lap_rows(47, stops, caps)
        assert len(rows) == _count_rows(47, stops, caps)
        lens = _lens_from_pit_laps(rows, 47)
        for _ in range(6):
            shape_modes = tuple(rng.choice(modes) for _ in range(stops + 1))
            tires = tuple(rng.random() < 0.6 for _ in range(stops))
            total, ok, tail1, tail2 = _shape_times(model, lens, shape_modes, tires)
            for i in rng.sample(range(len(lens)), min(5, len(lens))):
                spec = PlanSpec(lens=tuple(int(x) for x in lens[i]), modes=shape_modes,
                                tires=tires)
                plan = _evaluate(model, spec)
                assert plan.time_s == pytest.approx(total[i], abs=1e-6)
                assert plan.feasible == bool(ok[i])
                if spec.lens[-1] >= 3:
                    assert _evaluate(model, spec, upto=46).time_s == pytest.approx(
                        total[i] - tail1[i], abs=1e-6)
                    assert _evaluate(model, spec, upto=45).time_s == pytest.approx(
                        total[i] - tail2[i], abs=1e-6)


def test_best_plan_is_the_true_optimum_of_its_family():
    fit = make_fit(base=100.0, slope=0.06, weight=0.03, ref=80.0,
                   modes={"push": 3.0, "fuel-save@2.7": 2.7}, costs={"fuel-save@2.7": 0.6},
                   pit_fixed=27.0, rate=2.6, overhead=1.5, tire_time=20.0)
    race = Race(tank_l=70.0, laps=40)
    model = _Model(fit, race, State())
    result = solve(race, fit)
    one_stop = next(p for p in result.plans if p.spec.stops == 1 and all(p.spec.tires))
    brute = min(
        (plan.time_s for plan in (
            _evaluate(model, PlanSpec(lens=(a, 40 - a), modes=modes, tires=(True,)))
            for a in range(1, 40)
            for modes in itertools.product(fit.modes, repeat=2)) if plan.feasible))
    assert one_stop.time_s == pytest.approx(brute, abs=1e-6)


def test_coarse_grid_search_finds_the_same_plan(monkeypatch):
    import strategist.solver as solver
    fit = make_fit(base=100.0, slope=0.06, weight=0.03, ref=80.0,
                   modes={"push": 3.0, "fuel-save@2.7": 2.7}, costs={"fuel-save@2.7": 0.6},
                   pit_fixed=27.0, rate=2.6, overhead=1.5, tire_time=20.0)
    race = Race(tank_l=100.0, laps=80)
    exact = solve(race, fit)
    monkeypatch.setattr(solver, "EVAL_BUDGET", 2000)
    coarse = solve(race, fit)
    assert any("grid" in note for note in coarse.notes)
    assert coarse.plans[0].spec == exact.plans[0].spec
    assert coarse.plans[0].time_s == pytest.approx(exact.plans[0].time_s)


# -- the core question: save and drop a stop, or push and take it ------------------------

def test_saving_fuel_drops_a_stop_when_it_pays():
    # 34 laps: push needs 103.5 L (one stop), fuel-save needs 93.2 L (no stop).
    fit = make_fit(base=100.0, modes={"push": 3.0, "fuel-save@2.7": 2.7},
                   costs={"fuel-save@2.7": 0.4}, pit_fixed=27.0, rate=2.6, se=0.01)
    result = solve(Race(tank_l=100.0, laps=34), fit)
    assert result.plans[0].title == "No stop, fuel-save@2.7"
    push = next(p for p in result.plans if p.spec.stops == 1)
    assert push.title == "1 stop, push, no tire change"
    # Saving costs 34 x 0.4 s; the stop costs 27 s plus the fuel time.
    assert push.gap_s == pytest.approx(27.0 + push.stops[0].add_l / 2.6 - 34 * 0.4, abs=1e-6)
    assert result.plans[0].verdict == "clear winner"


def test_pushing_wins_when_saving_costs_too_much():
    fit = make_fit(base=100.0, modes={"push": 3.0, "fuel-save@2.7": 2.7},
                   costs={"fuel-save@2.7": 1.5}, pit_fixed=27.0, rate=2.6, se=0.01)
    result = solve(Race(tank_l=100.0, laps=34), fit)
    assert result.plans[0].title == "1 stop, push, no tire change"
    assert result.plans[1].title == "No stop, fuel-save@2.7"
    assert result.plans[1].gap_s > 0


def test_plans_inside_the_scatter_are_too_close_to_call():
    fit = make_fit(base=100.0, modes={"push": 3.0, "fuel-save@2.7": 2.7},
                   costs={"fuel-save@2.7": 0.9}, pit_fixed=27.0, rate=2.6, se=0.2, pit_se=1.0)
    result = solve(Race(tank_l=100.0, laps=34), fit)
    second = result.plans[1]
    assert abs(second.gap_s) < second.gap_u_s
    assert second.verdict == "too close to call"
    assert "too close to call" in result.plans[0].verdict
    assert "clear winner" not in render(result, Race(tank_l=100.0, laps=34), fit)


def test_the_list_is_never_padded():
    # Only one stop count works and saving has nothing to offer.
    fit = make_fit(base=100.0, modes={"push": 3.0}, pit_fixed=27.0, rate=2.6)
    result = solve(Race(tank_l=100.0, laps=60, max_stops=1), fit)
    assert len(result.plans) == 1
    assert result.plans[0].verdict == "only plan that works"
    many = solve(Race(tank_l=100.0, laps=60), make_fit(
        base=100.0, slope=0.05, weight=0.03, modes={"push": 3.0}, pit_fixed=27.0, rate=2.6,
        tire_time=20.0))
    assert 1 <= len(many.plans) <= 5
    assert [p.gap_s for p in many.plans] == sorted(p.gap_s for p in many.plans)


def test_gap_uncertainty_comes_from_the_parameter_spread():
    fit = make_fit(base=100.0, modes={"push": 3.0, "fuel-save@2.7": 2.7},
                   costs={"fuel-save@2.7": 0.4}, pit_fixed=27.0, rate=2.6, se=0.05, pit_se=0.8)
    result = solve(Race(tank_l=100.0, laps=34), fit)
    push = next(p for p in result.plans if p.spec.stops == 1)
    # The plans share base and differ by 34 laps of mode cost and one stop.
    assert push.gap_u_s == pytest.approx(math.hypot(34 * 0.05, 0.8))


# -- timed races -------------------------------------------------------------------------

def test_timed_race_counts_laps_per_plan_and_fuels_one_lap_more():
    fit = make_fit(base=100.0, modes={"push": 3.0, "fuel-save@2.7": 2.7},
                   costs={"fuel-save@2.7": 0.5}, pit_fixed=27.0, rate=2.6)
    # 56.5 minutes = 3390 s. Fuel-save laps take 100.5 s: lap 34 ends at 3417 s,
    # after the clock, so that plan runs 34 laps. The push plan loses about 30 s
    # in the pits and also ends on lap 34.
    race = Race(tank_l=100.0, minutes=56.5)
    result = solve(race, fit)
    save = next(p for p in result.plans if p.spec.stops == 0)
    assert save.laps == 34
    assert save.time_s == pytest.approx(34 * 100.5)
    assert save.time_s - 100.5 < 3390 <= save.time_s
    # Fuel covers one lap more than predicted plus the half-lap reserve.
    assert save.start_fuel_l == pytest.approx(2.7 * (34 + 1 + 0.5))
    push = next(p for p in result.plans if p.spec.stops == 1)
    total = push.start_fuel_l + sum(s.add_l for s in push.stops)
    assert total == pytest.approx(3.0 * (push.laps + 1 + 0.5))


def test_timed_plans_with_different_lap_counts_are_ranked_on_common_laps():
    fit = make_fit(base=100.0, modes={"push": 3.0, "fuel-save@2.7": 2.7},
                   costs={"fuel-save@2.7": 0.5}, pit_fixed=27.0, rate=2.6)
    # At 57 minutes the no-stop saver ends lap 34 at 3417 s (inside the clock)
    # and runs 35 laps; the stopper is still on lap 34 when the clock runs out.
    result = solve(Race(tank_l=100.0, minutes=57.0), fit)
    by_stops = {p.spec.stops: p for p in result.plans}
    assert by_stops[0].laps == 35 and by_stops[1].laps == 34
    assert result.plans[0] is by_stops[0]
    # The gap is measured when both have done 34 laps.
    assert by_stops[1].gap_s == pytest.approx(by_stops[1].time_s - 34 * 100.5)


def test_white_flag_rule_adds_a_lap():
    fit = make_fit(base=100.0, modes={"push": 3.0}, pit_fixed=27.0, rate=2.6)
    plain = solve(Race(tank_l=200.0, minutes=30.0), fit)
    white = solve(Race(tank_l=200.0, minutes=30.0, white_flag_lap=True), fit)
    assert plain.plans[0].laps == 18          # 1800 s: lap 18 is under way at zero
    assert white.plans[0].laps == 19
    assert white.start_fuel_l == pytest.approx(3.0 * (19 + 1.5))


# -- series rules ----------------------------------------------------------------------------

def test_mandatory_stop_is_respected():
    fit = make_fit(base=100.0, modes={"push": 3.0}, pit_fixed=27.0, rate=2.6)
    result = solve(Race(tank_l=100.0, laps=20, mandatory_stops=1), fit)
    assert result.kind == "plans"
    assert all(p.spec.stops >= 1 for p in result.plans)


def test_tire_set_limit_and_no_tire_changes():
    fit = make_fit(base=100.0, slope=0.08, weight=0.03, modes={"push": 3.0}, pit_fixed=27.0,
                   rate=2.6, tire_time=5.0)
    free = solve(Race(tank_l=100.0, laps=70), fit)
    assert all(free.plans[0].spec.tires)
    limited = solve(Race(tank_l=100.0, laps=70, tire_sets=2), fit)
    assert all(sum(p.spec.tires) <= 1 for p in limited.plans)
    none = solve(Race(tank_l=100.0, laps=70, tire_change_allowed=False), fit)
    assert all(not any(p.spec.tires) for p in none.plans)
    assert none.plans[0].time_s > limited.plans[0].time_s > free.plans[0].time_s


def test_sequential_service_makes_a_tire_stop_longer():
    kwargs = dict(base=100.0, slope=0.05, weight=0.03, modes={"push": 3.0}, pit_fixed=27.0,
                  rate=2.6, overhead=1.0, tire_time=12.0)
    together = solve(Race(tank_l=100.0, laps=50), make_fit(**kwargs))
    sequential = solve(Race(tank_l=100.0, laps=50, service_together=False),
                       make_fit(together=False, **kwargs))
    stop_t = next(p for p in together.plans if all(p.spec.tires)).stops[0]
    stop_s = next(p for p in sequential.plans if p.spec.stops == 1 and all(p.spec.tires)).stops[0]
    assert stop_t.stationary_s == pytest.approx(max(1.0 + stop_t.add_l / 2.6, 12.0))
    assert stop_s.stationary_s == pytest.approx(1.0 + stop_s.add_l / 2.6 + 12.0)


def test_one_fuel_load_fit_flags_plans_that_keep_tires(store):
    fit = fitted(store, practice_script())
    assert not fit.fuel_weight_fitted
    result = solve(Race(tank_l=100.0, laps=66, service_together=False), fit)
    assert result.kind == "plans"
    kept = [p for p in result.plans if not all(p.spec.tires)]
    assert kept
    assert all(any("extrapolated" in w for w in p.warnings) for p in kept)
    assert all(not any("extrapolated" in w for w in p.warnings)
               for p in result.plans if all(p.spec.tires))


def test_fuel_only_stop_and_one_fuel_load_still_gives_plans():
    # No tire change was ever timed: plans keep the tires and say so.
    fit = make_fit(base=100.0, slope=-0.02, modes={"push": 3.0}, pit_fixed=27.0, rate=2.6)
    result = solve(Race(tank_l=100.0, laps=60), fit)
    assert result.kind == "plans"
    assert all(not any(p.spec.tires) for p in result.plans)
    assert any("only plans that keep the tires" in note for note in result.notes)


def test_one_fuel_load_fit_flags_a_ranking_that_fuel_weight_could_flip(store):
    fit = fitted(store, practice_script())
    race = Race(tank_l=100.0, laps=36)
    result = solve(race, fit)
    assert result.plans[0].title == "No stop, fuel-save@2.7"
    rival = next(p for p in result.plans if p.title == "1 stop, push")
    assert rival.flip_weight is not None and rival.flip_weight < 0.06
    assert result.open_decisions and "fuel-weight" in result.open_decisions[0]
    assert "clear winner" not in render(result, race, fit)
    # Even stints, not the artefact of a stop pushed as late as it will go.
    assert abs(rival.spec.lens[0] - rival.spec.lens[1]) <= 6
    # With the second fuel load the question is settled, in favour of the truth.
    settled = solve(race, fitted(store, practice_script(second_load=55.0)))
    assert settled.plans[0].title == "1 stop, push" and not settled.open_decisions


# -- adjusting a plan -------------------------------------------------------------------------

@pytest.fixture
def two_stop():
    fit = make_fit(base=100.0, slope=0.05, weight=0.03, ref=80.0,
                   modes={"push": 3.0, "fuel-save@2.7": 2.7}, costs={"fuel-save@2.7": 0.6},
                   pit_fixed=27.0, rate=2.6, overhead=1.5, tire_time=20.0, se=0.02)
    race = Race(tank_l=100.0, laps=66)
    return race, fit, solve(race, fit)


def test_rescoring_an_untouched_plan_changes_nothing(two_stop):
    race, fit, result = two_stop
    best = result.plans[0]
    same = rescore(race, fit, best, reference=best)
    assert same.spec.lens == best.spec.lens
    assert same.time_s == pytest.approx(best.time_s)
    assert same.gap_s == pytest.approx(0.0) and same.gap_u_s == pytest.approx(0.0)


def test_locking_a_pit_lap_rescores_the_plan(two_stop):
    race, fit, result = two_stop
    best = result.plans[0]
    moved_to = best.stops[0].lap - 4
    moved = rescore(race, fit, best, pit_laps={1: moved_to}, reference=best)
    assert moved.feasible
    assert moved.stops[0].lap == moved_to and moved.stops[1].lap == best.stops[1].lap
    assert moved.gap_s > 0                       # the solver's plan was the optimum
    model = _Model(fit, race, State())
    direct = _evaluate(model, PlanSpec(lens=moved.spec.lens, modes=best.spec.modes,
                                       tires=best.spec.tires))
    assert moved.time_s == pytest.approx(direct.time_s)
    assert "adjusted" in render_adjusted(moved, 1)


def test_changing_a_fuel_amount_rescores_the_plan(two_stop):
    race, fit, result = two_stop
    best = result.plans[0]
    more = rescore(race, fit, best, fuel={1: best.stops[0].add_l + 15.0}, reference=best)
    assert more.feasible
    assert more.stops[0].add_l == pytest.approx(best.stops[0].add_l + 15.0)
    # The second stop tops up 15 L less; the extra weight costs time.
    assert more.stops[1].add_l == pytest.approx(best.stops[1].add_l - 15.0)
    assert more.gap_s > 0
    short = rescore(race, fit, best, fuel={1: best.stops[0].add_l - 20.0}, reference=best)
    assert not short.feasible and "runs out of fuel on lap" in short.problems[0]
    too_much = rescore(race, fit, best, fuel={0: 130.0}, reference=best)
    assert not too_much.feasible and "tank" in too_much.problems[0]


def test_switching_a_pace_mode_rescores_the_plan(two_stop):
    race, fit, result = two_stop
    best = result.plans[0]
    saved = rescore(race, fit, best, modes={2: "fuel-save@2.7"}, reference=best)
    assert saved.feasible and saved.spec.modes[1] == "fuel-save@2.7"
    laps = best.spec.lens[1]
    assert saved.gap_s > 0
    assert saved.stops[0].add_l < best.stops[0].add_l      # the stint needs less fuel
    unknown = rescore(race, fit, best, modes={2: "cruise"}, reference=best)
    assert not unknown.feasible and "not been measured" in unknown.problems[0]
    no_tires = rescore(race, fit, best, tires={1: False}, reference=best)
    assert no_tires.feasible and no_tires.spec.tires == (False, True)


def test_adjusting_a_timed_plan_recomputes_the_lap_count():
    fit = make_fit(base=100.0, modes={"push": 3.0, "fuel-save@2.7": 2.7},
                   costs={"fuel-save@2.7": 2.0}, pit_fixed=27.0, rate=2.6)
    race = Race(tank_l=100.0, minutes=100.0)
    result = solve(race, fit)
    best = next(p for p in result.plans if set(p.spec.modes) == {"push"})
    slow = rescore(race, fit, best, modes={1: "fuel-save@2.7", 2: "fuel-save@2.7"},
                   reference=best)
    assert slow.feasible and slow.laps == best.laps - 1
    assert slow.time_s - 102.0 < 6000 <= slow.time_s


# -- from the current state (what the live re-plan will call) ---------------------------------

def test_solving_from_the_current_state():
    fit = make_fit(base=100.0, slope=0.05, weight=0.03, ref=80.0, modes={"push": 3.0},
                   pit_fixed=27.0, rate=2.6, overhead=1.5, tire_time=20.0)
    race = Race(tank_l=100.0, laps=60)
    state = State(laps_done=20, fuel_l=30.0, tire_age=20, elapsed_s=2050.0, stops_done=0)
    result = solve(race, fit, state)
    assert result.kind == "plans"
    best = result.plans[0]
    assert best.laps == 60 and best.stints[0].first_lap == 21
    # 30 L lasts 9 laps with the reserve in hand, so the stop comes by lap 29.
    assert 21 <= best.stops[0].lap <= 29
    assert best.stints[0].fuel_start_l == pytest.approx(30.0)
    stranded = solve(race, fit, State(laps_done=58, fuel_l=2.0, tire_age=30))
    assert stranded.kind == "no-plan"


# -- the whole chain against the simulator ------------------------------------------------------

def test_predicted_race_matches_a_simulated_race(store):
    truth = SimTruth()
    fit = fitted(store, practice_script(second_load=55.0), seed=7)
    race = Race(tank_l=truth.tank_l, laps=40)
    best = solve(race, fit).plans[0]
    script = [Grid(best.start_fuel_l)]
    for stint, stop in itertools.zip_longest(best.stints, best.stops):
        script.append(Run(stint.mode.split("@")[0], stint.last_lap - stint.first_lap + 1))
        if stop:
            script.append(Stop(add_l=stop.add_l, tires=stop.tires))
    sid = log_session(store, truth, script, session_type="Race", race_laps=40, seed=99)
    laps = store.laps_df([sid])
    assert len(laps) == 40
    driven = laps["session_time_s"].iloc[-1] - (laps["session_time_s"].iloc[0]
                                                - laps["lap_time_s"].iloc[0])
    # Lap 1 starts from the grid in the simulator; compare from the first line crossing.
    assert driven == pytest.approx(best.time_s, abs=4 * best.gap_u_s + 0.03 * 40 * 5)
    # The reserve is intact at the flag.
    assert laps["fuel_at_line_l"].iloc[-1] == pytest.approx(0.5 * 3.0, abs=0.6)


def test_solver_ranks_like_the_truth_when_the_fit_is_good(store):
    truth = SimTruth()
    fit = fitted(store, practice_script(second_load=55.0), seed=7)
    exact = truth_fit(truth)
    for laps in (36, 50, 66):
        race = Race(tank_l=truth.tank_l, laps=laps)
        got, want = solve(race, fit), solve(race, exact)
        assert got.plans[0].title == want.plans[0].title
        model = _Model(exact, race, State())
        for plan in got.plans[1:]:
            true_gap = (_evaluate(model, plan.spec).time_s
                        - _evaluate(model, got.plans[0].spec).time_s)
            assert abs(plan.gap_s - true_gap) <= 3.5 * plan.gap_u_s + 0.5
