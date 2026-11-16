import pytest

from conftest import log_session, practice_script
from strategist.logger import LapLogger, verify_session
from strategist.sim import Grid, Reset, Run, Simulator, SimTruth, Stop
from strategist.telemetry import Sample


def test_practice_stint_logs_every_lap(store):
    truth = SimTruth()
    sid = log_session(store, truth, [Reset(100.0), Run("push", 12)])
    laps = store.laps_df([sid])
    assert laps["lap_number"].tolist() == list(range(1, 13))
    assert verify_session(store, sid) == []
    # Lap 1 leaves the pits, the rest are flying laps under the push tag.
    assert laps.iloc[0]["pit_road"] == "out" and not laps.iloc[0]["clean"]
    assert laps.iloc[1:]["clean"].all()
    assert (laps["stint_tag"] == "push").all()
    assert laps["tire_age"].tolist() == list(range(12))
    assert laps["lap_in_stint"].tolist() == list(range(1, 13))
    # iRacing's own lap time is used once it shows up.
    assert (laps.iloc[1:]["lap_time_source"] == "sim").all()
    assert laps.iloc[1:]["fuel_used_l"].between(2.9, 3.1).all()


def test_lap_times_match_the_simulated_truth(store):
    truth = SimTruth(noise_sd=0.0, fuel_noise_sd=0.0)
    sid = log_session(store, truth, [Reset(100.0), Run("push", 6)])
    laps = store.laps_df([sid])
    fuel = laps["fuel_start_l"].to_numpy()
    age = laps["tire_age"].to_numpy()
    expected = truth.base + 0.05 * age + truth.fuel_weight * fuel
    got = laps["lap_time_s"].to_numpy()
    assert got[1:] == pytest.approx(expected[1:], abs=1e-3)


def test_timer_fallback_when_sim_gives_no_lap_time(store):
    truth = SimTruth(noise_sd=0.0)
    sim = Simulator(truth, [Reset(100.0), Run("push", 4)], hz=60)
    ids = []
    logger = LapLogger(store=store, info_provider=lambda n: sim.info, source="sim",
                       on_session=lambda sid, info: ids.append(sid))
    for event in sim.events():
        if isinstance(event, Sample):
            event.lap_last_time = -1.0
            logger.process(event)
    logger.finalize()
    laps = store.laps_df(ids)
    assert (laps["lap_time_source"] == "timer").all()
    expected = truth.base + 0.05 * laps["tire_age"] + truth.fuel_weight * laps["fuel_start_l"]
    assert laps["lap_time_s"].to_numpy()[1:] == pytest.approx(expected.to_numpy()[1:], abs=0.04)


def test_pit_stop_record_matches_the_stopwatch(store):
    truth = SimTruth()
    sid = log_session(store, truth, [
        Reset(60.0), Run("push", 6), Stop(add_l=40.0, tires=True), Run("push", 5)])
    pits = store.pits_df([sid])
    assert len(pits) == 1
    stop = pits.iloc[0]
    stationary = truth.stationary_time(40.0, True)
    assert stop["lap"] == 6
    assert stop["stationary_time_s"] == pytest.approx(stationary, abs=0.15)
    assert stop["pit_lane_time_s"] == pytest.approx(truth.lane_drive_s + stationary, abs=0.15)
    assert stop["fuel_added_l"] == pytest.approx(40.0, abs=0.05)
    assert stop["fuel_time_s"] == pytest.approx(40.0 / truth.refuel_rate, abs=0.15)
    assert stop["tires_changed"] == 1
    assert stop["entry_pct"] == pytest.approx(truth.pit_entry_pct, abs=0.01)
    assert stop["exit_pct"] == pytest.approx(truth.pit_exit_pct, abs=0.01)

    laps = store.laps_df([sid])
    assert laps["lap_number"].tolist() == list(range(1, 12))
    assert verify_session(store, sid) == []
    in_lap, out_lap = laps.iloc[5], laps.iloc[6]
    assert in_lap["pit_road"] == "in" and out_lap["pit_road"] == "out"
    assert in_lap["fuel_added_l"] == pytest.approx(40.0, abs=0.05)
    assert in_lap["fuel_used_l"] == pytest.approx(3.0, abs=0.1)
    # The stop splits the stints and resets the tire age on the out-lap.
    assert laps["stint_number"].tolist() == [1] * 6 + [2] * 5
    assert laps["tire_age"].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
    assert laps["lap_in_stint"].tolist() == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5]


def test_fuel_only_stop_keeps_the_tire_age(store):
    sid = log_session(store, SimTruth(), [
        Reset(30.0), Run("push", 4), Stop(add_l=20.0, tires=False), Run("push", 3)])
    pits = store.pits_df([sid])
    assert pits.iloc[0]["tires_changed"] == 0
    laps = store.laps_df([sid])
    assert laps["tire_age"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert laps["stint_number"].tolist() == [1, 1, 1, 1, 2, 2, 2]


def test_reset_to_pits_starts_a_new_stint_without_gaps(store):
    sid = log_session(store, SimTruth(), practice_script())
    laps = store.laps_df([sid])
    assert verify_session(store, sid) == []
    assert sorted(laps["stint_number"].unique()) == [1, 2, 3, 4]
    save = laps[laps["stint_tag"] == "fuel-save"]
    assert (save["tag_target"] == 2.7).all()
    assert save["clean"].sum() == 12
    # The lap abandoned at each reset is kept as an unfinished row.
    resets = laps[laps["unclean_reason"] == "reset"]
    assert len(resets) == 2 and resets["lap_time_s"].isna().all()
    # Every stint starts on new tires.
    assert (laps.groupby("stint_number")["tire_age"].min() == 0).all()


def test_incident_and_traffic_laps_are_not_clean(store):
    truth = SimTruth()
    sim = Simulator(truth, [Reset(100.0), Run("push", 8, incident_laps=(4,))])
    ids = []
    logger = LapLogger(store=store, info_provider=lambda n: sim.info, source="sim",
                       on_session=lambda sid, info: ids.append(sid))
    marked = False
    for event in sim.events():
        if isinstance(event, Sample):
            logger.process(event)
            if event.lap == 6 and not marked:
                assert logger.mark_traffic() == 6
                marked = True
        else:
            logger.set_tag(event.tag, event.target)
    logger.finalize()
    laps = store.laps_df(ids).set_index("lap_number")
    assert laps.loc[4, "incidents"] == 1 and laps.loc[4, "unclean_reason"] == "incident"
    assert laps.loc[6, "traffic"] == 1 and laps.loc[6, "clean"] == 1
    assert laps.loc[5, "clean"] == 1 and laps.loc[5, "traffic"] == 0


def test_tag_change_mid_lap_spoils_that_lap(store):
    sim = Simulator(SimTruth(), [Reset(100.0), Run("push", 6)])
    ids = []
    logger = LapLogger(store=store, info_provider=lambda n: sim.info, source="sim",
                       on_session=lambda sid, info: ids.append(sid))
    changed = False
    for event in sim.events():
        if isinstance(event, Sample):
            logger.process(event)
            if event.lap == 4 and event.lap_dist_pct > 0.5 and not changed:
                logger.set_tag("fuel-save", 2.7)
                changed = True
        else:
            logger.set_tag(event.tag, event.target)
    logger.finalize()
    laps = store.laps_df(ids).set_index("lap_number")
    assert laps.loc[3, "stint_tag"] == "push" and laps.loc[3, "clean"] == 1
    assert laps.loc[4, "stint_tag"] == "fuel-save"
    assert laps.loc[4, "unclean_reason"] == "tag-change"
    assert laps.loc[5, "stint_tag"] == "fuel-save" and laps.loc[5, "clean"] == 1


def test_race_is_logged_with_stops(store):
    truth = SimTruth()
    sid = log_session(store, truth, [
        Grid(60.0), Run("push", 15), Stop(add_l=50.0, tires=True), Run("push", 15)],
        session_type="Race", race_laps=30)
    laps = store.laps_df([sid])
    assert laps["lap_number"].tolist() == list(range(1, 31))
    assert verify_session(store, sid) == []
    assert laps.iloc[0]["unclean_reason"] == "race-start"
    assert store.session(sid)["race_laps"] == 30
    assert len(store.pits_df([sid])) == 1
    assert laps["clean"].sum() == 30 - 3      # start lap, in-lap, out-lap


def test_unknown_tag_is_rejected(store):
    logger = LapLogger(store=store, info_provider=lambda n: None)
    with pytest.raises(ValueError):
        logger.set_tag("cruise")
