import json
import math

import pytest

from conftest import log_session, practice_script
from strategist.fitter import MIN_MODE_LAPS, Fit, compare_fits, fit_parameters
from strategist.sim import Reset, Run, SimMode, SimTruth, Stop


def fit_session(store, script, truth=None, seed=1, **kwargs):
    truth = truth or SimTruth()
    sid = log_session(store, truth, script, seed=seed)
    fit = fit_parameters(store.laps_df([sid]), store.pits_df([sid]), car=truth.car,
                         track=truth.track, tank_l=truth.tank_l, **kwargs)
    return fit, sid


def within(param, truth_value, sigmas=3.0):
    assert param is not None and param.status == "ok", param
    return abs(param.value - truth_value) <= sigmas * param.spread


def test_recovers_the_simulated_truth_with_two_fuel_loads(store):
    truth = SimTruth()
    fit, _ = fit_session(store, practice_script(second_load=55.0), seed=2)
    assert within(fit.param("fuel_per_lap", "push"), 3.0)
    assert within(fit.param("fuel_per_lap", "fuel-save@2.7"), 2.7)
    assert fit.fuel_weight_fitted
    assert within(fit.param("fuel_weight"), truth.fuel_weight)
    assert within(fit.param("tire_age_loss"), 0.05)
    assert within(fit.param("mode_cost", "fuel-save@2.7"), 0.6)
    base = truth.base + truth.fuel_weight * fit.ref_fuel
    assert within(fit.param("base_lap_time"), base)
    assert fit.resid_sd == pytest.approx(truth.noise_sd, rel=0.35)


def test_declines_fuel_weight_with_one_fuel_load(store):
    fit, _ = fit_session(store, practice_script())
    weight = fit.param("fuel_weight")
    assert weight.status == "declined" and weight.value is None
    assert "one fuel load" in weight.note
    assert not fit.fuel_weight_fitted and "fuel" not in fit.columns
    # The stint slope is still reported, labelled as the net of fuel burn-off.
    slope = fit.param("tire_age_loss")
    assert slope.status == "ok" and "net of fuel burn-off" in slope.note
    assert within(slope, 0.05 - 0.03 * 3.0, sigmas=4.0)


def test_pit_stop_parameters(store):
    truth = SimTruth()
    fit, _ = fit_session(store, practice_script())
    pit = fit.pit
    lane_truth = truth.lane_drive_s - truth.lane_pct * fit.coef("base")
    assert pit.n_stops == 1
    assert pit.fixed_loss == pytest.approx(lane_truth + truth.out_lap_penalty, abs=0.6)
    assert pit.lane_loss == pytest.approx(lane_truth, abs=0.2)
    assert pit.out_lap_penalty == pytest.approx(truth.out_lap_penalty, abs=0.6)
    assert pit.refuel_rate == pytest.approx(truth.refuel_rate, rel=0.03)
    assert pit.tire_time == pytest.approx(truth.tire_time, abs=0.2)
    assert pit.tire_time_kind == "measured"
    assert pit.overhead == pytest.approx(truth.service_overhead, abs=0.15)


def test_tire_time_is_an_upper_bound_when_hidden_behind_fuel(store):
    truth = SimTruth(tire_time=8.0)
    script = [Reset(100.0), Run("push", 12),
              Reset(60.0), Run("push", 6), Stop(add_l=40.0, tires=True), Run("push", 5)]
    fit, _ = fit_session(store, script, truth=truth)
    assert fit.pit.tire_time_kind == "upper-bound"
    assert "upper bound" in fit.param("tire_change_time").note
    assert fit.pit.tire_time >= truth.tire_time


def test_fuel_only_stop_declines_tire_time(store):
    script = [Reset(100.0), Run("push", 12),
              Reset(60.0), Run("push", 6), Stop(add_l=30.0, tires=False), Run("push", 5)]
    fit, _ = fit_session(store, script)
    assert fit.param("tire_change_time").status == "declined"
    assert fit.pit.tire_time is None
    assert fit.pit.overhead == pytest.approx(SimTruth().service_overhead, abs=0.15)
    assert "fuel-only" in fit.pit.overhead_note


def test_sequential_service_tire_time(store):
    truth = SimTruth(service_together=False, tire_time=12.0)
    script = [Reset(100.0), Run("push", 12),
              Reset(60.0), Run("push", 6), Stop(add_l=20.0, tires=False), Run("push", 6),
              Stop(add_l=20.0, tires=True), Run("push", 5)]
    fit, _ = fit_session(store, script, truth=truth, service_together=False)
    assert fit.pit.n_stops == 2
    assert fit.pit.tire_time == pytest.approx(12.0, abs=0.3)
    assert fit.pit.tire_time_kind == "measured"


def test_two_sessions_agree_within_their_spread(store):
    a, _ = fit_session(store, practice_script(second_load=55.0), seed=11)
    b, _ = fit_session(store, practice_script(second_load=55.0), seed=12)
    rows = compare_fits(a, b)
    names = {(r.name, r.mode) for r in rows}
    assert ("fuel_per_lap", "push") in names and ("fuel_weight", None) in names
    judged = [r for r in rows if r.agree is not None]
    assert len(judged) >= 7
    assert all(r.agree for r in judged), [r for r in judged if not r.agree]


def test_sessions_that_differ_are_caught(store):
    a, _ = fit_session(store, practice_script(), seed=3)
    thirsty = SimTruth(modes={"push": SimMode(3.3, 0.0, 0.05),
                              "fuel-save": SimMode(2.7, 0.6, 0.05)})
    b, _ = fit_session(store, practice_script(), truth=thirsty, seed=4)
    verdict = {(r.name, r.mode): r.agree for r in compare_fits(a, b)}
    assert verdict[("fuel_per_lap", "push")] is False
    assert verdict[("fuel_per_lap", "fuel-save@2.7")] is True


def test_drops_out_laps_in_laps_incident_and_traffic_laps(store):
    truth = SimTruth()
    script = [Reset(100.0), Run("push", 14, incident_laps=(5,), slow_laps=(9,)),
              Reset(100.0), Run("push", 6), Stop(add_l=18.0, tires=True), Run("push", 5)]
    sid = log_session(store, truth, script)
    # Lap 9 was held up; without the traffic tag the slow lap is still in the data.
    store.mark_traffic(sid, 9)
    laps = store.laps_df([sid])
    fit = fit_parameters(laps, store.pits_df([sid]), car=truth.car, track=truth.track,
                         tank_l=truth.tank_l)
    # 14 laps minus out-lap, incident lap, traffic lap; 11 laps minus two out-laps and an in-lap.
    assert fit.modes["push"].n_laps == (14 - 3) + (11 - 3)
    assert fit.resid_sd == pytest.approx(truth.noise_sd, rel=0.4)
    assert fit.notes == []


def test_untagged_slow_lap_is_rejected_as_an_outlier(store):
    truth = SimTruth()
    sid = log_session(store, truth, [Reset(100.0), Run("push", 16, slow_laps=(9,))])
    fit = fit_parameters(store.laps_df([sid]), store.pits_df([sid]), car=truth.car,
                         track=truth.track, tank_l=truth.tank_l)
    assert any("left out" in note for note in fit.notes)
    assert fit.resid_sd == pytest.approx(truth.noise_sd, rel=0.4)


def test_mode_with_too_few_laps_is_declined(store):
    script = [Reset(100.0), Run("push", 12), Reset(100.0), Run("fuel-save", 5, target=2.7)]
    fit, _ = fit_session(store, script)
    assert "fuel-save@2.7" not in fit.modes
    declined = fit.param("fuel_per_lap", "fuel-save@2.7")
    assert declined.status == "declined" and str(MIN_MODE_LAPS) in declined.note
    assert fit.param("mode_cost", "fuel-save@2.7") is None


def test_no_push_stint_means_no_lap_model(store):
    fit, _ = fit_session(store, [Reset(100.0), Run("fuel-save", 12, target=2.7)])
    assert not fit.has_lap_model
    assert fit.param("base_lap_time").status == "declined"
    assert fit.param("pit_lane_loss").status == "declined"


def test_tire_save_gets_its_own_wear_rate(store):
    script = practice_script(second_load=55.0) + [Reset(100.0), Run("tire-save", 16)]
    fit, _ = fit_session(store, script)
    own = fit.param("tire_age_loss", "tire-save")
    assert fit.modes["tire-save"].slope_group == "tire-save"
    assert within(own, 0.015)
    assert own.value < fit.param("tire_age_loss").value


def test_fit_round_trips_through_the_store(store):
    fit, _ = fit_session(store, practice_script(second_load=55.0))
    fit_id = store.save_fit(fit)
    loaded = Fit.from_dict(store.latest_fit_json(fit.car, fit.track))
    assert loaded.to_dict() == json.loads(json.dumps(fit.to_dict()))
    rows = store.db.execute("SELECT * FROM parameters WHERE fit_id=?", (fit_id,)).fetchall()
    assert len(rows) == len(fit.params)
    assert {r["status"] for r in rows} == {"ok"}
    assert not math.isnan(loaded.coef("base"))
