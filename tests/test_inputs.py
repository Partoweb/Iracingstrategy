import pytest

from strategist.inputs import parse_hotkeys, parse_key, parse_line


def test_typed_lines():
    assert parse_line("p") == ("tag", {"tag": "push", "target": None})
    assert parse_line("f 2.6") == ("tag", {"tag": "fuel-save", "target": 2.6})
    assert parse_line("fuel-save 2,55") == ("tag", {"tag": "fuel-save", "target": 2.55})
    assert parse_line("t") == ("tag", {"tag": "tire-save", "target": None})
    assert parse_line("x") == ("traffic", {"last": False})
    assert parse_line("xl") == ("traffic", {"last": True})
    assert parse_line("x last") == ("traffic", {"last": True})
    assert parse_line("") is None
    assert parse_line("banana") is None
    assert parse_line("f lots") is None


def test_hotkey_specs():
    assert parse_key("F13") == (0, 0x7C)
    assert parse_key("ctrl+alt+p") == (0x0003, ord("P"))
    assert parse_hotkeys("push=F13, fuel-save=F14,traffic=shift+F16") == {
        "push": (0, 0x7C), "fuel-save": (0, 0x7D), "traffic": (0x0004, 0x7F)}
    with pytest.raises(ValueError):
        parse_hotkeys("cruise=F13")
    with pytest.raises(ValueError):
        parse_key("ctrl+banana")
