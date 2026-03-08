from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from typing import TYPE_CHECKING
from modules.inpainting.lama import LaMa
from modules.inpainting.mi_gan import MIGAN
from modules.inpainting.aot import AOT
from modules.inpainting.schema import Config
from app.ui.messages import Messages
from app.ui.settings.settings_page import SettingsPage

if TYPE_CHECKING:
    from controller import ComicTranslate

inpaint_map = {
    "LaMa": LaMa,
    "MI-GAN": MIGAN,
    "AOT": AOT,
}

REMOTE_OCR_TOOLS = {"Gemini-2.0-Flash", "Microsoft OCR", "Google Cloud Vision", "GPT-4.1-mini"}

OCR_CREDENTIAL_REQUIREMENTS = {
    "Microsoft OCR": ("Microsoft Azure", ("api_key_ocr", "endpoint"), "an OCR API key and endpoint URL"),
    "Google Cloud Vision": ("Google Cloud", ("api_key",), "an API key"),
    "GPT-4.1-mini": ("Open AI GPT", ("api_key",), "an API key"),
    "Gemini-2.0-Flash": ("Google Gemini", ("api_key",), "an API key"),
}

def get_config(settings_page: SettingsPage):
    strategy_settings = settings_page.get_hd_strategy_settings()
    if strategy_settings['strategy'] == settings_page.ui.tr("Resize"):
        config = Config(hd_strategy="Resize", hd_strategy_resize_limit = strategy_settings['resize_limit'])
    elif strategy_settings['strategy'] == settings_page.ui.tr("Crop"):
        config = Config(hd_strategy="Crop", hd_strategy_crop_margin = strategy_settings['crop_margin'],
                        hd_strategy_crop_trigger_size = strategy_settings['crop_trigger_size'])
    else:
        config = Config(hd_strategy="Original")

    return config

def validate_ocr(main: ComicTranslate):
    """Ensure OCR tool selection and any required credentials are configured."""
    settings_page = main.settings_page
    settings = settings_page.get_all_settings()
    ocr_tool = settings['tools']['ocr']
    credentials = settings.get('credentials', {})

    if not ocr_tool:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Text Recognition model"))
        return False

    if ocr_tool not in REMOTE_OCR_TOOLS:
        return True

    service_name, required_fields, requirement_text = OCR_CREDENTIAL_REQUIREMENTS.get(ocr_tool, (None, (), "credentials"))
    if not service_name:
        return True

    service_credentials = credentials.get(service_name, {})
    if not all(service_credentials.get(field) for field in required_fields):
        Messages.show_service_not_configured_error(main, ocr_tool, requirement_text)
        return False

    return True


def validate_translator(main: ComicTranslate, target_lang: str):
    """Ensure either API credentials are set or the user is authenticated, plus check compatibility."""
    settings_page = main.settings_page
    tr = settings_page.ui.tr
    settings = settings_page.get_all_settings()
    credentials = settings.get('credentials', {})
    translator_tool = settings['tools']['translator']

    if not translator_tool:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Translator"))
        return False

    # Credential checks
    if "Custom" in translator_tool:
        # Custom requires api_key, api_url, and model to be configured LOCALLY
        service = tr('Custom')
        creds = credentials.get(service, {})
        # Check if all required fields are present and non-empty
        if not all([creds.get('api_key'), creds.get('api_url'), creds.get('model')]):
            Messages.show_custom_not_configured_error(main)
            return False
        return True

    if not settings_page.is_logged_in():
        Messages.show_not_logged_in_error(main)
        return False

    return True

def font_selected(main: ComicTranslate):
    if not main.render_settings().font_family:
        Messages.select_font_error(main)
        return False
    return True

def validate_settings(main: ComicTranslate, target_lang: str):
    if not validate_ocr(main):
        return False
    if not validate_translator(main, target_lang):
        return False
    if not font_selected(main):
        return False
    
    return True
