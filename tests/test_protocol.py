from conftest import log_session, practice_script
from strategist.fitter import fit_parameters
from strategist.protocol import render, save_target, stint_progress, stint_protocol
from strategist.sim import Reset, Run, SimTruth, Stop
from strategist.solver import Race, solve


def protocol_for(store, script, race=None):
    truth = SimTruth()
    sid = log_session(store, truth, script) if script else None
    ids = [sid] if sid else []
    laps = store.laps_df(ids)
    fit = fit_parameters(laps, store.pits_df(ids), car=truth.car, track=truth.track,
                         tank_l=truth.tank_l)
    result = solve(race, fit) if race else None
    return stint_protocol(laps, fit, tank_l=truth.tank_l, race=race, result=result), fit


def test_nothing_logged_asks_for_the_push_stint(store):
    proto, _ = protocol_for(store, None)
    assert [s.done for s in proto.steps[:3]] == [False, False, False]
    assert proto.next_text.startswith("Drive a push stint")
    assert "10 clean laps" in proto.next_text and "100 L" in proto.next_text


def test_after_the_push_stint_it_states_a_fuel_save_target(store):
    race = Race(tank_l=100.0, laps=36)
    proto, _ = protocol_for(store, [Reset(100.0), Run("push", 13)], race=race)
    assert proto.steps[0].done and not proto.steps[1].done
    # 36 laps on push needs 109.5 L: one stop. No stop needs 100 / 36.5 = 2.74 L per lap.
    assert "Drive a fuel-save stint" in proto.next_text
    assert "holding 2.73 L per lap" in proto.next_text
    assert "turns 1 stop into 0" in proto.next_text
    assert "fuel-save 2.73" in proto.next_text


def test_save_target_without_a_race_and_when_out_of_reach():
    target, why = save_target(3.0, 100.0, None, 100.0)
    assert target == 2.85 and "5%" in why
    # 60 laps: dropping from one stop to none would need 1.65 L per lap.
    target, why = save_target(3.0, 100.0, Race(tank_l=100.0, laps=60), 100.0)
    assert target == 2.85 and "out of reach" in why
    target, why = save_target(3.0, 100.0, Race(tank_l=100.0, laps=20), 100.0)
    assert target == 2.85 and "no stop" in why
    # Timed: 60 min at 100 s is 37 laps, 38.5 with the margin: 100 / 38.5 = 2.597.
    target, why = save_target(3.0, 100.0, Race(tank_l=100.0, minutes=60.0), 100.0)
    assert target == 2.59 and "turns 1 stop into 0" in why


def test_short_stint_does_not_count(store):
    proto, _ = protocol_for(store, [Reset(100.0), Run("push", 7)])
    assert not proto.steps[0].done
    assert "6 clean laps in the longest stint" in proto.steps[0].detail


def test_then_it_asks_for_the_timed_stop(store):
    script = [Reset(100.0), Run("push", 13), Reset(100.0), Run("fuel-save", 13, target=2.7)]
    proto, _ = protocol_for(store, script)
    assert [s.done for s in proto.steps[:3]] == [True, True, False]
    assert proto.next_text.startswith("Make one pit stop")


def test_third_stint_only_when_an_open_decision_needs_it(store):
    # 36 laps: saving and stopping are close enough for fuel weight to decide.
    proto, fit = protocol_for(store, practice_script(), race=Race(tank_l=100.0, laps=36))
    assert not fit.fuel_weight_fitted
    step = proto.steps[3]
    assert step.title == "Second fuel load" and not step.done
    assert "Drive one more push stint from about 60 L" in proto.next_text
    assert "[ ] Second fuel load" in render(proto)


def test_no_third_stint_when_nothing_is_open(store):
    # 25 laps needs no stop on push: nothing to decide.
    proto, _ = protocol_for(store, practice_script(), race=Race(tank_l=100.0, laps=25))
    assert proto.complete
    assert proto.steps[3].title == "Third stint" and "not needed" in proto.steps[3].detail
    assert "Nothing more to drive" in proto.next_text


def test_second_fuel_load_closes_the_protocol(store):
    proto, fit = protocol_for(store, practice_script(second_load=55.0),
                              race=Race(tank_l=100.0, laps=36))
    assert fit.fuel_weight_fitted and proto.complete
    assert proto.steps[3].detail == "fuel-weight effect measured"


def test_stint_progress_line(store):
    truth = SimTruth()
    sid = log_session(store, truth, [Reset(100.0), Run("push", 13),
                                     Stop(add_l=30.0, tires=True), Run("push", 4)])
    laps = store.laps_df([sid])
    assert stint_progress(laps.iloc[:6], sid, 1) == "push stint: 5/10 clean laps, 5 to go"
    assert "enough" in stint_progress(laps.iloc[:11], sid, 1)
    assert stint_progress(laps.iloc[:12], sid, 1) == ""
    assert stint_progress(laps, sid, 2) == "push stint: 3/10 clean laps, 7 to go"
