"""Stint-tag input for a running logger: typed lines and global hotkeys."""

from __future__ import annotations

import queue
import sys
import threading
from typing import Any

from .logger import TAGS

Command = tuple[str, dict[str, Any]]

ALIASES = {"p": "push", "f": "fuel-save", "t": "tire-save",
           "push": "push", "fuel-save": "fuel-save", "tire-save": "tire-save",
           "fuel": "fuel-save", "tire": "tire-save", "tyre-save": "tire-save"}

HELP = ("type and press Enter:  p = push   f [target] = fuel-save   t [target] = tire-save   "
        "x = this lap had traffic   xl = last lap had traffic")


def parse_line(line: str) -> Command | None:
    """Turn a typed line such as 'f 2.6' into a logger command."""
    words = line.strip().lower().split()
    if not words:
        return None
    head = words[0]
    if head in ("x", "traffic"):
        return ("traffic", {"last": len(words) > 1 and words[1] in ("l", "last")})
    if head in ("xl",):
        return ("traffic", {"last": True})
    if head in ALIASES:
        target = None
        if len(words) > 1:
            try:
                target = float(words[1].replace(",", "."))
            except ValueError:
                return None
        return ("tag", {"tag": ALIASES[head], "target": target})
    return None


class ConsoleInput:
    """Reads typed lines from the logger's own window without blocking the loop."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            for line in sys.stdin:
                self._queue.put(line)
        except (OSError, ValueError):
            pass

    def poll(self) -> list[Command]:
        commands: list[Command] = []
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                return commands
            command = parse_line(line)
            if command:
                commands.append(command)
            elif line.strip():
                print(f"not understood: {line.strip()!r}.  {HELP}")


# -- global hotkeys (Windows) ---------------------------------------------------

MODIFIERS = {"alt": 0x0001, "ctrl": 0x0002, "control": 0x0002, "shift": 0x0004, "win": 0x0008}
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
NAMED_KEYS = {"space": 0x20, "pause": 0x13, "insert": 0x2D, "delete": 0x2E, "home": 0x24,
              "end": 0x23, "pageup": 0x21, "pagedown": 0x22}
ACTIONS = (*TAGS, "traffic")


def parse_key(spec: str) -> tuple[int, int]:
    """Parse 'F13' or 'ctrl+alt+p' into (modifier bits, virtual-key code)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty key")
    mods = 0
    for part in parts[:-1]:
        if part not in MODIFIERS:
            raise ValueError(f"unknown modifier {part!r} in {spec!r}")
        mods |= MODIFIERS[part]
    key = parts[-1]
    if key in NAMED_KEYS:
        return mods, NAMED_KEYS[key]
    if key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        return mods, 0x70 + int(key[1:]) - 1
    if len(key) == 1 and (key.isalpha() or key.isdigit()):
        return mods, ord(key.upper())
    raise ValueError(f"unknown key {key!r} in {spec!r}")


def parse_hotkeys(spec: str) -> dict[str, tuple[int, int]]:
    """Parse 'push=F13,fuel-save=F14,traffic=ctrl+alt+x' into {action: (mods, vk)}."""
    mapping: dict[str, tuple[int, int]] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        action, _, key = item.partition("=")
        action = action.strip().lower()
        if action not in ACTIONS:
            raise ValueError(f"unknown hotkey action {action!r}; use one of {', '.join(ACTIONS)}")
        mapping[action] = parse_key(key)
    if not mapping:
        raise ValueError("no hotkeys given")
    return mapping


class GlobalHotkeys:
    """System-wide hotkeys, so a wheel button mapped to a key works while the sim has focus.

    Uses the Win32 RegisterHotKey API through ctypes; Windows only.
    """

    def __init__(self, mapping: dict[str, tuple[int, int]]):
        if sys.platform != "win32":
            raise RuntimeError("global hotkeys need Windows")
        self._mapping = mapping
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._errors: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:  # pragma: no cover - needs Windows
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        actions = list(self._mapping)
        for index, action in enumerate(actions, start=1):
            mods, vk = self._mapping[action]
            if not user32.RegisterHotKey(None, index, mods | MOD_NOREPEAT, vk):
                self._errors.put(f"could not register the hotkey for {action} "
                                 "(another program may already use it)")
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and 1 <= msg.wParam <= len(actions):
                self._queue.put(actions[msg.wParam - 1])

    def poll(self) -> list[Command]:
        while not self._errors.empty():
            print(self._errors.get_nowait())
        commands: list[Command] = []
        while True:
            try:
                action = self._queue.get_nowait()
            except queue.Empty:
                return commands
            if action == "traffic":
                commands.append(("traffic", {"last": False}))
            else:
                commands.append(("tag", {"tag": action, "target": None}))
