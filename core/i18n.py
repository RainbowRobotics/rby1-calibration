import os
import yaml
from PySide6.QtCore import QObject, Signal

class I18nManager(QObject):
    """
    Centralized Internationalization (i18n) Manager for dynamic language switching.
    Emits language_changed signal when user toggles between English and Korean.
    """
    language_changed = Signal(str)

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = I18nManager()
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_lang = "en"  # Default language: English ("en")
        self.translations = {}
        self.load_translations()

    def load_translations(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "i18n.yaml")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.translations = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[I18nManager] Failed to load translations from {config_path}: {e}")
        else:
            print(f"[I18nManager] Translation file not found: {config_path}")

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

    def get(self, key, lang=None, default=None):
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
                return default if default is not None else key

        if isinstance(val, dict):
            if lang in val:
                return val[lang]
            elif "en" in val:
                return val["en"]
            elif "ko" in val:
                return val["ko"]
        
        return default if default is not None else str(val)

# Helper function
def tr(key, lang=None, default=None):
    return I18nManager.instance().get(key, lang=lang, default=default)
