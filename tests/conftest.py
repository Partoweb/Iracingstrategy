import pytest

from strategist.logger import LapLogger
from strategist.sim import Grid, Reset, Run, Simulator, SimTruth, Stop, feed
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
