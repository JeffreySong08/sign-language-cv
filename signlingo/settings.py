"""Three persistent camera preferences, separate from bundled training data."""

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import sys


@dataclass
class CameraSettings:
    low_power: bool = False
    show_landmarks: bool = True
    selfie: bool = True


def settings_path():
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "SignLingo" / "settings.json"


class SettingsStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else settings_path()
        self.values = CameraSettings()
        self.revision = 0
        self.error = ""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in asdict(self.values):
                    if type(data.get(key)) is bool:
                        setattr(self.values, key, data[key])
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            logging.warning("Could not load camera settings; using defaults", exc_info=True)

    def toggle(self, key):
        if key not in asdict(self.values):
            raise ValueError(f"Unknown camera preference: {key}")
        setattr(self.values, key, not getattr(self.values, key))
        self.revision += 1
        self.save()

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(asdict(self.values), indent=2), encoding="utf-8")
            temporary.replace(self.path)
            self.error = ""
        except OSError:
            self.error = "Could not save. Changes apply for this session only."
            logging.exception("Could not save camera settings")


_store = None


def get_settings():
    global _store
    if _store is None:
        _store = SettingsStore()
    return _store
