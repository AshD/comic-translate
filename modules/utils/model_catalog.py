from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List, Tuple

from .download import ModelID

DETECTOR_CHOICES = [
    "RT-DETR-v2",
]

OCR_CHOICES = [
    "Default",
    "Manga OCR",
    "Pororo OCR",
    "PP-OCRv5 English",
    "PP-OCRv5 Chinese",
    "PP-OCRv5 Korean",
    "PP-OCRv5 Latin",
    "PP-OCRv5 Russian",
    "PP-OCRv5 Server (Chinese)",
    "GPT-4.1-mini",
    "Google Cloud Vision",
    "Gemini-2.0-Flash",
    "Microsoft OCR",
]

INPAINTER_CHOICES = [
    "LaMa",
    "MI-GAN",
    "AOT",
]

DETECTOR_HELP = {
    "RT-DETR-v2": "Recommended. Comic-specific bubble and text detector. Best default choice in this repo.",
}

OCR_HELP = {
    "Default": "Recommended. Picks a local OCR tuned for the current source language: Manga OCR for Japanese, Pororo for Korean, and PP-OCRv5 for most others.",
    "Manga OCR": "Best local choice for Japanese manga text. Use when the source language is Japanese.",
    "Pororo OCR": "Best local choice for Korean text in this repo. Use when the source language is Korean.",
    "PP-OCRv5 English": "Local OCR tuned for English. Fast and practical for Latin-script comics.",
    "PP-OCRv5 Chinese": "Local OCR for Simplified and Traditional Chinese pages.",
    "PP-OCRv5 Korean": "Local PaddleOCR Korean recognizer. Good fallback if Pororo misses text.",
    "PP-OCRv5 Latin": "Local OCR for French, German, Spanish, Italian, Dutch, and other Latin-script pages.",
    "PP-OCRv5 Russian": "Local OCR for Russian and East Slavic text.",
    "PP-OCRv5 Server (Chinese)": "Heavier Chinese PP-OCRv5 model. Slower but can help on difficult Chinese pages.",
    "GPT-4.1-mini": "Hosted OCR through OpenAI vision. Useful for difficult pages, but slower and requires API credentials.",
    "Google Cloud Vision": "Hosted OCR from Google Cloud Vision. Useful if local OCR struggles.",
    "Gemini-2.0-Flash": "Hosted OCR using Gemini vision. Can help on hard layouts and stylized text.",
    "Microsoft OCR": "Hosted OCR using Azure AI Vision. Good general-purpose fallback if local OCR fails.",
}

INPAINTER_HELP = {
    "LaMa": "Recommended for most pages. Good quality and stable local inpainting.",
    "MI-GAN": "Alternative local inpainter. Useful if LaMa or AOT leaves artifacts on some pages.",
    "AOT": "Fast local inpainter and a good default for large batch jobs.",
}

DEFAULT_OCR_BY_SOURCE = {
    "Japanese": [ModelID.MANGA_OCR_BASE_ONNX],
    "Korean": [ModelID.PORORO_ONNX],
    "Chinese": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_MOBILE],
    "Russian": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_ESLAV_MOBILE],
    "French": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_LATIN_MOBILE],
    "English": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_EN_MOBILE],
    "Spanish": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_LATIN_MOBILE],
    "Italian": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_LATIN_MOBILE],
    "German": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_LATIN_MOBILE],
    "Dutch": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_LATIN_MOBILE],
}

LOCAL_OCR_MODEL_IDS = {
    "Manga OCR": [ModelID.MANGA_OCR_BASE_ONNX],
    "Pororo OCR": [ModelID.PORORO_ONNX],
    "PP-OCRv5 English": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_EN_MOBILE],
    "PP-OCRv5 Chinese": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_MOBILE],
    "PP-OCRv5 Korean": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_KOREAN_MOBILE],
    "PP-OCRv5 Latin": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_LATIN_MOBILE],
    "PP-OCRv5 Russian": [ModelID.PPOCR_V5_DET_MOBILE, ModelID.PPOCR_V5_REC_ESLAV_MOBILE],
    "PP-OCRv5 Server (Chinese)": [ModelID.PPOCR_V5_DET_SERVER, ModelID.PPOCR_V5_REC_SERVER],
}

LOCAL_INPAINTER_MODEL_IDS = {
    "LaMa": [ModelID.LAMA_ONNX],
    "MI-GAN": [ModelID.MIGAN_PIPELINE_ONNX],
    "AOT": [ModelID.AOT_ONNX],
}

LOCAL_DETECTOR_MODEL_IDS = {
    "RT-DETR-v2": [ModelID.RTDETR_V2_ONNX],
}

HOSTED_OCR_CHOICES = {
    "GPT-4.1-mini",
    "Google Cloud Vision",
    "Gemini-2.0-Flash",
    "Microsoft OCR",
}


def _unique(values: Iterable[ModelID]) -> list[ModelID]:
    return list(OrderedDict((value, None) for value in values).keys())


def get_detector_help(choice: str) -> str:
    return DETECTOR_HELP.get(choice, "")


def get_ocr_help(choice: str) -> str:
    return OCR_HELP.get(choice, "")


def get_inpainter_help(choice: str) -> str:
    return INPAINTER_HELP.get(choice, "")


def get_local_model_download_plan(
    detector_key: str,
    ocr_key: str,
    inpainter_key: str,
    source_lang_english: str,
) -> Tuple[List[ModelID], List[str]]:
    model_ids: list[ModelID] = []
    notes: list[str] = []

    model_ids.extend(LOCAL_DETECTOR_MODEL_IDS.get(detector_key, []))

    if ocr_key == "Default":
        chosen = DEFAULT_OCR_BY_SOURCE.get(source_lang_english, DEFAULT_OCR_BY_SOURCE["English"])
        model_ids.extend(chosen)
        notes.append(f"Default OCR resolves to local models for {source_lang_english}.")
    elif ocr_key in LOCAL_OCR_MODEL_IDS:
        model_ids.extend(LOCAL_OCR_MODEL_IDS[ocr_key])
    elif ocr_key in HOSTED_OCR_CHOICES:
        notes.append(f"{ocr_key} is a hosted OCR service and has no local model files to download.")

    model_ids.extend(LOCAL_INPAINTER_MODEL_IDS.get(inpainter_key, []))
    return _unique(model_ids), notes
