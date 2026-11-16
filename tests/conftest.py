import pytest

from strategist.fitter import Fit, ModeFit, PitFit
from strategist.logger import LapLogger
from strategist.sim import Reset, Run, Simulator, Stop, feed
from strategist.store import Store


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "test.db") as s:
        yield s


def log_session(store, truth, script, **kwargs):
    """Run a scripted session through the logger. Returns the session id."""
    sim = Simulator(truth, script, **kwargs)
    ids = []
    logger = LapLogger(store=store, info_provider=lambda num: sim.info, source="sim",
                       on_session=lambda sid, info: ids.append(sid))
    feed(logger, sim)
    assert len(ids) == 1
    return ids[0]


def practice_script(push_laps=13, save_laps=13, target=2.7, second_load=None):
    """The default protocol: push stint, fuel-save stint, then a timed stop.

    Every stint starts from a full tank. `second_load` adds a push stint from
    that many litres, which is what the fuel-weight effect needs.
    """
    script = [
        Reset(100.0), Run("push", push_laps),
        Reset(100.0), Run("fuel-save", save_laps, target=target),
        Reset(100.0), Run("push", 6), Stop(add_l=18.0, tires=True), Run("push", 5),
    ]
    if second_load is not None:
        script += [Reset(second_load), Run("push", 12)]
    return script


def make_fit(*, base=100.0, slope=0.0, weight=None, ref=50.0, modes=None, costs=None,
             pit_fixed=25.0, rate=2.5, overhead=0.0, tire_time=None, together=True,
             se=0.0, pit_se=0.0, max_age=40, tire_save_slope=None):
    """A hand-made fit with exact parameters, for checking the solver by hand."""
    modes = modes or {"push": 3.0}
    costs = costs or {}
    columns = ["base", "wear:default"]
    theta = [base, slope]
    mode_fits = {}
    for name, burn in modes.items():
        tag = name.split("@")[0]
        group = name if tag == "tire-save" else "default"
        mode_fits[name] = ModeFit(name=name, tag=tag, target=None, fuel_per_lap=burn,
                                  fuel_sd=0.02, n_laps=12, max_age=max_age, slope_group=group)
        if group != "default":
            columns.append(f"wear:{name}")
            theta.append(tire_save_slope if tire_save_slope is not None else slope)
    for name in modes:
        if name != "push":
            columns.append(f"cost:{name}")
            theta.append(costs.get(name, 0.0))
    if weight is not None:
        columns.append("fuel")
        theta.append(weight)
    cov = [[(se ** 2 if i == j else 0.0) for j in range(len(columns))]
           for i in range(len(columns))]
    pit = None
    if pit_fixed is not None:
        pit = PitFit(n_stops=1, fixed_loss=pit_fixed, fixed_loss_se=pit_se, lane_loss=None,
                     out_lap_penalty=None, refuel_rate=rate, refuel_rate_se=None,
                     overhead=overhead, overhead_note="", tire_time=tire_time,
                     tire_time_se=None,
                     tire_time_kind="measured" if tire_time is not None else None,
                     service_together=together)
    return Fit(car="Test car", track="Test track", session_ids=[1], tank_l=None,
               modes=mode_fits, columns=columns, theta=theta, cov=cov, resid_sd=0.1,
               ref_fuel=ref, fuel_weight_fitted=weight is not None, pit=pit, n_clean_laps=24)


def truth_fit(truth, target=2.7, ref=90.0):
    """The simulator's own parameters, written as a fit with no uncertainty."""
    save = f"fuel-save@{target:g}"
    lap = truth.base + truth.fuel_weight * ref
    return make_fit(
        base=lap, slope=truth.modes["push"].deg, weight=truth.fuel_weight, ref=ref,
        modes={"push": truth.modes["push"].fuel_per_lap,
               save: truth.modes["fuel-save"].fuel_per_lap},
        costs={save: truth.modes["fuel-save"].cost},
        pit_fixed=truth.lane_drive_s - truth.lane_pct * lap + truth.out_lap_penalty,
        rate=truth.refuel_rate, overhead=truth.service_overhead, tire_time=truth.tire_time,
        together=truth.service_together)
