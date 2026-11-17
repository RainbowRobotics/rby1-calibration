from core.storage import ConfigStorage
from core.storage import FileStorage
import os
import yaml
from PySide6.QtCore import QObject, Signal
from core.storage import CONFIG_PATHS

DEFAULT_UI_DROPDOWNS = {
    "robot_models": ["a", "m"],
    "arm_sides": ["Right Arm", "Left Arm"],
    "marker_axes": ["Axis 6 (Yaw Sweep, ±20°)", "Axis 5 (Pitch Sweep, ±10°)"],
    "joint_modes_v13": ["wrist_roll_v13 (6-Axis Sweep)", "wrist_pitch_v13 (5-Axis Sweep)", "elbow (3-Axis Sweep)"],
    "joint_modes_v12": ["wrist_yaw2 (6-Axis Sweep)", "wrist_pitch (5-Axis Sweep)", "elbow (3-Axis Sweep)"]
}

_CACHED_STYLESHEET = None


def load_ui_dropdowns(config_path=None) -> dict:
    """Load dropdown choices from ui_dropdowns.yaml with fallback to defaults."""
    if config_path is None:
        config_path = CONFIG_PATHS.get("ui_dropdowns_yaml")

    if config_path and os.path.exists(config_path):
        try:
            with FileStorage.open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"[Language] Warning: Failed to load ui_dropdowns from {config_path}: {e}")

    return DEFAULT_UI_DROPDOWNS.copy()


def load_stylesheet(config_path=None, reload=False) -> str:
    """Load dark QSS stylesheet from config/ui_config/dark_theme.qss."""
    global _CACHED_STYLESHEET
    if _CACHED_STYLESHEET is not None and not reload:
        return _CACHED_STYLESHEET

    if config_path is None:
        config_path = CONFIG_PATHS.get("dark_theme_qss")

    if config_path and os.path.exists(config_path):
        try:
            with FileStorage.open(config_path, "r", encoding="utf-8") as f:
                _CACHED_STYLESHEET = f.read()
                return _CACHED_STYLESHEET
        except Exception as e:
            print(f"[Language] Warning: Failed to load stylesheet from {config_path}: {e}")

    return ""


# Global loaded constants for direct import
UI_DROPDOWNS = load_ui_dropdowns()
DARK_STYLESHEET = load_stylesheet()


class LanguageManager(QObject):
    """
    Centralized Language & Localization Manager for dynamic language switching.
    Emits language_changed signal when user toggles between English and Korean.
    """
    language_changed = Signal(str)

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = LanguageManager()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_lang = "en"  # Default language: English ("en")
        self.translations = {}
        self.load_translations()

    def load_translations(self, config_path=None):
        if config_path is None:
            config_path = CONFIG_PATHS.get("i18n_yaml")
            # Fallback check for old location if needed
            if not config_path or not os.path.exists(config_path):
                old_path = os.path.join(os.path.dirname(__file__), "..", "config", "i18n.yaml")
                if os.path.exists(old_path):
                    config_path = old_path

        if config_path and os.path.exists(config_path):
            try:
                self.translations = ConfigStorage.load(config_path)
            except Exception as e:
                print(f"[LanguageManager] Failed to load translations from {config_path}: {e}")
        else:
            print(f"[LanguageManager] Translation file not found: {config_path}")
        self._fill_from_bundle(config_path)

    def _fill_from_bundle(self, config_path):
        """A frozen build only copies config files that are missing next to the exe, so an
        i18n.yaml left there by an older build lacks every key added since and the UI would show
        raw key names. Keys the external file does not have are taken from the bundled copy."""
        import sys
        if not getattr(sys, "frozen", False) or not hasattr(sys, "_MEIPASS"):
            return
        bundled = os.path.join(sys._MEIPASS, "config", "ui_config", "i18n.yaml")
        if not os.path.exists(bundled) or (config_path and os.path.abspath(bundled) == os.path.abspath(config_path)):
            return
        try:
            base = ConfigStorage.load(bundled) or {}
        except Exception:
            return

        def fill(target, source):
            for key, value in source.items():
                if key not in target:
                    target[key] = value
                elif isinstance(target[key], dict) and isinstance(value, dict):
                    fill(target[key], value)

        if not isinstance(self.translations, dict):
            self.translations = {}
        fill(self.translations, base)

    def set_language(self, lang):
        lang = lang.lower()
        if lang in ("ko", "korean"):
            lang_code = "ko"
        elif lang in ("en", "english"):
            lang_code = "en"
        else:
            lang_code = "ko"

        if self.current_lang != lang_code:
            self.current_lang = lang_code
            self.language_changed.emit(self.current_lang)

    @property
    def is_korean(self) -> bool:
        return self.current_lang == "ko"

    def get(self, key, lang=None, default=None, **kwargs):
        if lang is None:
            lang = self.current_lang
        else:
            lang = "ko" if lang in ("ko", "korean") else "en"

        parts = key.split(".")
        val = self.translations
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                res = default if default is not None else key
                if kwargs and isinstance(res, str):
                    try:
                        return res.format(**kwargs)
                    except Exception:
                        return res
                return res

        res = None
        if isinstance(val, dict):
            if lang in val:
                res = val[lang]
            elif "en" in val:
                res = val["en"]
            elif "ko" in val:
                res = val["ko"]

        if res is None:
            res = default if default is not None else str(val)

        if kwargs and isinstance(res, str):
            try:
                return res.format(**kwargs)
            except Exception:
                return res
        return res


# Helper function
def tr(key, lang=None, default=None, **kwargs):
    return LanguageManager.instance().get(key, lang=lang, default=default, **kwargs)
