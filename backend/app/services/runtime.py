"""Runtime switches an operator can flip without restarting (persisted in data/runtime.json)."""

import json
import threading

from app.config import get_settings

_lock = threading.Lock()


def _path():
    return get_settings().storage_dir.parent / "runtime.json"  # next to the database, so tests stay isolated


def _read() -> dict:
    try:
        return json.loads(_path().read_text())
    except Exception:
        return {}


def online_requested() -> bool:
    """The operator's choice (default OFF: nothing leaves the machine until someone opts in)."""
    return bool(_read().get("online_enabled", False))


def set_online(enabled: bool) -> None:
    with _lock:
        data = _read()
        data["online_enabled"] = bool(enabled)
        _path().parent.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(data))


def online_available() -> bool:
    """Online reading can be used right now: a key exists and the hard offline switch is not set."""
    s = get_settings()
    return bool(s.nvidia_api_key) and not s.offline_mode


def online_active() -> bool:
    return online_available() and online_requested()


def status() -> dict:
    s = get_settings()
    return {
        "provider": "NVIDIA NIM",
        "model": s.nvidia_model,
        "configured": bool(s.nvidia_api_key),
        "offline_mode": s.offline_mode,
        "enabled": online_requested(),
        "active": online_active(),
        "note": "Page images are sent to NVIDIA's servers when active. Switch off to stay fully offline.",
    }
