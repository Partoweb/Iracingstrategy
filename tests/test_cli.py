import pytest

from strategist.cli import main


@pytest.fixture
def db(tmp_path, capsys):
    path = str(tmp_path / "cli.db")
    assert main(["--db", path, "demo"]) == 0
    capsys.readouterr()
    return path


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_sessions_laps_and_verify(db, capsys):
    code, out, _ = run(capsys, "--db", db, "sessions")
    assert code == 0 and "Practice" in out and "Race" in out and "Sim GT3 · Sim Ring" in out
    code, out, _ = run(capsys, "--db", db, "laps", "1")
    assert code == 0 and "fuel-save@2.7" in out and "pit stop on lap" in out
    for session in ("1", "2"):
        code, out, _ = run(capsys, "--db", db, "verify", session)
        assert code == 0 and "none missing" in out
    code, _, err = run(capsys, "--db", db, "verify", "9")
    assert code == 2 and "no session 9" in err


def test_fit_saves_parameters_and_declines_fuel_weight(db, capsys):
    code, out, _ = run(capsys, "--db", db, "fit")
    assert code == 0
    assert "fuel_per_lap" in out and "saved as fit 1" in out
    weight = next(line for line in out.splitlines() if line.startswith("fuel_weight"))
    assert "declined" in weight and "one fuel load" in weight


def test_stint_screen(db, capsys):
    code, out, _ = run(capsys, "--db", db, "stint", "--laps", "36")
    assert code == 0
    assert "[x] Push stint" in out and "[ ] Second fuel load" in out
    assert "Next: Drive one more push stint" in out


def test_plan_view_and_adjustment(db, capsys):
    code, out, _ = run(capsys, "--db", db, "plan", "--laps", "50")
    assert code == 0
    assert "clear-air race time" in out and "Based on" in out
    assert "1. " in out and "Pit lap" in out and ", add " in out
    first_stop = next(line for line in out.splitlines() if line.strip().startswith("Pit lap"))
    lap = int(first_stop.split()[2].rstrip(","))
    code, out, _ = run(capsys, "--db", db, "plan", "--laps", "50", "--adjust", "1",
                       "--pit", f"1={lap - 3}")
    assert code == 0
    adjusted = out.split("Plan 1, adjusted")[1]
    assert f"Pit lap {lap - 3}," in adjusted and "[+" in adjusted.splitlines()[1]
    code, out, _ = run(capsys, "--db", db, "plan", "--laps", "50", "--fuel", "1=5")
    assert code == 0 and "does not work: runs out of fuel on lap" in out
    # Moving a stop so far that the next stint cannot be fuelled is refused with the reason.
    code, out, _ = run(capsys, "--db", db, "plan", "--laps", "50", "--pit", "1=5")
    assert code == 0 and "does not fit the 100 L tank" in out


def test_no_stop_race_and_missing_race_length(db, capsys):
    code, out, _ = run(capsys, "--db", db, "plan", "--laps", "20")
    assert code == 0 and "No stop needed. Start fuel:" in out and "Pit lap" not in out
    code, _, err = run(capsys, "--db", db, "plan")
    assert code == 2 and "race length" in err
    code, _, err = run(capsys, "--db", db, "plan", "--laps", "20", "--pit", "x")
    assert code == 2 and "no plan to adjust" in err


def test_race_file(db, capsys, tmp_path):
    race = tmp_path / "race.toml"
    race.write_text('minutes = 120\nmandatory_stops = 2\nservice = "sequential"\n')
    code, out, _ = run(capsys, "--db", db, "plan", "--race", str(race))
    assert code == 0 and "120 min" in out and "2 mandatory stops" in out
    assert "tires after fuel" in out
    race.write_text("lapz = 3\n")
    code, _, err = run(capsys, "--db", db, "plan", "--race", str(race))
    assert code == 2 and "unknown setting" in err


def test_compare_two_sessions(tmp_path, capsys):
    path = str(tmp_path / "two.db")
    assert main(["--db", path, "demo", "--seed", "11"]) == 0
    assert main(["--db", path, "demo", "--seed", "12"]) == 0
    capsys.readouterr()
    code, out, _ = run(capsys, "--db", path, "compare", "1", "3")
    assert code == 0, out
    assert "All parameters agree within their spread." in out
    assert "fuel weight declined" in out


def test_tag_and_traffic_commands(db, capsys):
    from strategist.store import Store

    code, out, _ = run(capsys, "--db", db, "tag", "fuel-save", "2.6")
    assert code == 0 and "fuel-save@2.6" in out
    code, out, _ = run(capsys, "--db", db, "traffic", "--session", "1", "--lap", "5")
    assert code == 0 and "tagged as traffic" in out
    with Store(db) as store:
        assert store.take_commands() == [("tag", {"tag": "fuel-save", "target": 2.6})]
        assert store.laps_df([1]).set_index("lap_number").loc[5, "traffic"] == 1


def test_empty_store_explains_what_to_do(tmp_path, capsys):
    code, _, err = run(capsys, "--db", str(tmp_path / "empty.db"), "plan", "--laps", "30")
    assert code == 2 and "Nothing logged yet" in err
