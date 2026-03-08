from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, List

import imkit as imk
import requests
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QColor

from app.path_materialization import ensure_path_materialized
from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from app.ui.messages import Messages
from modules.detection.processor import TextBlockDetector
from modules.rendering.render import get_best_render_area, is_vertical_block, pyside_word_wrap
from modules.translation.processor import Translator
from modules.utils.device import resolve_device
from modules.utils.exceptions import InsufficientCreditsException
from modules.utils.image_utils import generate_mask, get_smart_text_color
from modules.utils.language_utils import get_language_code, is_no_space_lang
from modules.utils.pipeline_config import get_config, inpaint_map
from modules.utils.textblock import sort_blk_list
from modules.utils.translator_utils import format_translations, get_raw_text, get_raw_translation
from .cache_manager import CacheManager
from .block_detection import BlockDetectionHandler
from .inpainting import InpaintingHandler
from .ocr_handler import OCRHandler

if TYPE_CHECKING:
    from controller import ComicTranslate

logger = logging.getLogger(__name__)


class _IdentityUI:
    @staticmethod
    def tr(text: str) -> str:
        return text


class _SettingsSnapshot:
    def __init__(self, payload: dict):
        self._payload = payload
        self.ui = _IdentityUI()

    @classmethod
    def from_settings_page(cls, settings_page):
        mappings = dict(getattr(settings_page.ui, 'value_mappings', {}))

        def normalize(value):
            if isinstance(value, dict):
                return {normalize(key): normalize(inner) for key, inner in value.items()}
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, str):
                return mappings.get(value, value)
            return value

        payload = normalize(settings_page.get_all_settings())
        return cls(payload)

    def get_tool_selection(self, tool_type: str):
        return self._payload['tools'][tool_type]

    def is_gpu_enabled(self):
        return bool(self._payload['tools']['use_gpu'])

    def get_llm_settings(self):
        return dict(self._payload['llm'])

    def get_export_settings(self):
        return dict(self._payload['export'])

    def get_hd_strategy_settings(self):
        return dict(self._payload['tools']['hd_strategy'])

    def get_credentials(self, service: str = ""):
        credentials = self._payload.get('credentials', {})
        if not service:
            return dict(credentials)
        return dict(credentials.get(service, {'save_key': self._payload.get('save_keys', False)}))

    def get_all_settings(self):
        return self._payload


class _BatchPageContext:
    def __init__(self, settings_page: _SettingsSnapshot, lang_mapping: dict):
        self.settings_page = settings_page
        self.lang_mapping = lang_mapping


@dataclass(frozen=True)
class _WorkItem:
    index: int
    image_path: str
    source_lang: str
    target_lang: str
    skip: bool
    directory: str
    archive_bname: str
    base_name: str
    extension: str


@dataclass
class _RenderEmitItem:
    text: str
    font_size: int
    block: object


@dataclass
class _PageResult:
    index: int
    image_path: str
    success: bool
    directory: str
    archive_bname: str
    base_name: str
    extension: str
    blk_list: list | None = None
    text_items_state: list | None = None
    patches: list | None = None
    render_emit_items: list | None = None
    skipped_stage: str = ""
    skipped_message: str = ""
    skip_reason: str = ""
    skip_traceback: str = ""


class BatchProcessor:
    """Handles batch processing of comic translation."""

    def __init__(
        self,
        main_page: ComicTranslate,
        cache_manager: CacheManager,
        block_detection_handler: BlockDetectionHandler,
        inpainting_handler: InpaintingHandler,
        ocr_handler: OCRHandler,
    ):
        self.main_page = main_page
        self.cache_manager = cache_manager
        self.block_detection = block_detection_handler
        self.inpainting = inpainting_handler
        self.ocr_handler = ocr_handler
        self._thread_local = None

    def skip_save(self, directory, timestamp, base_name, extension, archive_bname, image):
        logger.info("Skipping fallback translated image save for '%s'.", base_name)

    def emit_progress(self, index, total, step, steps, change_name):
        stage_map = {
            0: 'start-image',
            1: 'text-block-detection',
            2: 'ocr-processing',
            3: 'pre-inpaint-setup',
            4: 'generate-mask',
            5: 'inpainting',
            7: 'translation',
            9: 'text-rendering-prepare',
            10: 'save-and-finish',
        }
        stage_name = stage_map.get(step, f'stage-{step}')
        logger.info(
            "Progress: image_index=%s/%s step=%s/%s (%s) change_name=%s",
            index,
            total,
            step,
            steps,
            stage_name,
            change_name,
        )
        self.main_page.progress_update.emit(index, total, step, steps, change_name)

    def log_skipped_image(self, directory, timestamp, image_path, reason="", full_traceback=""):
        return

    def _is_cancelled(self) -> bool:
        worker = getattr(self.main_page, "current_worker", None)
        return bool(worker and worker.is_cancelled)

    def _get_thread_local_state(self):
        if self._thread_local is None:
            import threading
            self._thread_local = threading.local()
        return self._thread_local

    def _get_thread_local_detector(self, settings_snapshot: _SettingsSnapshot):
        state = self._get_thread_local_state()
        detector = getattr(state, 'detector', None)
        if detector is None:
            detector = TextBlockDetector(settings_snapshot)
            state.detector = detector
        return detector

    def _get_thread_local_ocr(self):
        state = self._get_thread_local_state()
        ocr = getattr(state, 'ocr', None)
        if ocr is None:
            from modules.ocr.processor import OCRProcessor
            ocr = OCRProcessor()
            state.ocr = ocr
        return ocr

    def _get_thread_local_inpainter(self, settings_snapshot: _SettingsSnapshot):
        state = self._get_thread_local_state()
        inpainter_key = settings_snapshot.get_tool_selection('inpainter')
        cached_key = getattr(state, 'inpainter_key', None)
        inpainter = getattr(state, 'inpainter', None)
        if inpainter is None or cached_key != inpainter_key:
            backend = 'onnx'
            device = resolve_device(settings_snapshot.is_gpu_enabled(), backend=backend)
            inpainter_class = inpaint_map[inpainter_key]
            logger.info("pre-inpaint: initializing inpainter '%s' on device %s", inpainter_key, device)
            t0 = time.time()
            inpainter = inpainter_class(device, backend=backend)
            state.inpainter = inpainter
            state.inpainter_key = inpainter_key
            logger.info("pre-inpaint: inpainter initialized in %.2fs", time.time() - t0)
        return inpainter

    def _build_archive_lookup(self):
        lookup = {}
        for archive in self.main_page.file_handler.archive_info:
            archive_path = archive['archive_path']
            directory = os.path.dirname(archive_path)
            archive_bname = os.path.splitext(os.path.basename(archive_path))[0].strip()
            for img_path in archive['extracted_images']:
                lookup[img_path] = (directory, archive_bname)
        return lookup

    def _build_work_items(self, image_list: list[str]):
        archive_lookup = self._build_archive_lookup()
        items = []
        for index, image_path in enumerate(image_list):
            state = self.main_page.image_states.get(image_path, {})
            source_lang = state.get('source_lang')
            target_lang = state.get('target_lang')
            base_name = os.path.splitext(os.path.basename(image_path))[0].strip()
            extension = os.path.splitext(image_path)[1]
            default_directory = os.path.dirname(image_path)
            directory, archive_bname = archive_lookup.get(image_path, (default_directory, ""))
            items.append(
                _WorkItem(
                    index=index,
                    image_path=image_path,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    skip=bool(state.get('skip', False)),
                    directory=directory,
                    archive_bname=archive_bname,
                    base_name=base_name,
                    extension=extension,
                )
            )
        return items

    def _extract_patches(self, mask, inpainted_image):
        contours, _ = imk.find_contours(mask)
        patches = []
        for contour in contours:
            x, y, w, h = imk.bounding_rect(contour)
            patch = inpainted_image[y:y + h, x:x + w]
            patches.append({'bbox': [x, y, w, h], 'image': patch.copy()})
        return patches

    def _format_pipeline_error(self, error: Exception, context: str) -> str:
        if isinstance(error, requests.exceptions.ConnectionError):
            return QCoreApplication.translate("Messages", "Unable to connect to the server.\nPlease check your internet connection.")
        if isinstance(error, requests.exceptions.HTTPError):
            status_code = error.response.status_code if error.response is not None else 500
            if status_code >= 500:
                return Messages.get_server_error_text(status_code, context=context)
            try:
                err_json = error.response.json()
                if "detail" in err_json and isinstance(err_json["detail"], dict):
                    return err_json["detail"].get("error_description", str(error))
                return err_json.get("error_description", str(error))
            except Exception:
                return str(error)
        return str(error)

    def _process_single_page(
        self,
        item: _WorkItem,
        timestamp: str,
        settings_snapshot: _SettingsSnapshot,
        context: _BatchPageContext,
        render_settings,
        button_to_alignment: dict,
    ) -> _PageResult:
        result = _PageResult(
            index=item.index,
            image_path=item.image_path,
            success=False,
            directory=item.directory,
            archive_bname=item.archive_bname,
            base_name=item.base_name,
            extension=item.extension,
            blk_list=[],
            text_items_state=[],
            patches=[],
            render_emit_items=[],
        )
        if self._is_cancelled():
            return result

        ensure_path_materialized(item.image_path)
        image = imk.read_image(item.image_path)
        if item.skip:
            result.skipped_stage = "User-skipped"
            result.skip_reason = "User-skipped"
            return result

        detector = self._get_thread_local_detector(settings_snapshot)
        blk_list = detector.detect(image)
        if not blk_list:
            result.skipped_stage = "Text Blocks"
            result.skip_reason = "No text blocks detected"
            return result

        source_lang = item.source_lang
        target_lang = item.target_lang
        settings_page = settings_snapshot

        ocr_model = settings_page.get_tool_selection('ocr')
        device = resolve_device(settings_page.is_gpu_enabled())
        cache_key = self.cache_manager._get_ocr_cache_key(image, source_lang, ocr_model, device)
        ocr_processor = self._get_thread_local_ocr()
        ocr_processor.initialize(context, source_lang)
        try:
            ocr_processor.process(image, blk_list)
            self.cache_manager._cache_ocr_results(cache_key, blk_list)
            source_lang_english = context.lang_mapping.get(source_lang, source_lang)
            rtl = source_lang_english == 'Japanese'
            blk_list = sort_blk_list(blk_list, rtl)
        except InsufficientCreditsException:
            raise
        except Exception as error:
            err_msg = self._format_pipeline_error(error, 'ocr')
            logger.exception("OCR processing failed: %s", err_msg)
            result.skipped_stage = "OCR"
            result.skipped_message = err_msg
            result.skip_reason = f"OCR: {err_msg}"
            result.skip_traceback = traceback.format_exc()
            return result

        export_settings = settings_page.get_export_settings()
        inpainter = self._get_thread_local_inpainter(settings_page)
        config = get_config(settings_page)
        logger.info("pre-inpaint: generating mask (blk_list=%d blocks)", len(blk_list))
        t0 = time.time()
        mask = generate_mask(image, blk_list)
        logger.info("pre-inpaint: mask generated in %.2fs (mask shape=%s)", time.time() - t0, getattr(mask, 'shape', None))
        inpaint_input_img = inpainter(image, mask, config)
        inpaint_input_img = imk.convert_scale_abs(inpaint_input_img)
        patches = self._extract_patches(mask, inpaint_input_img)
        result.patches = patches

        if export_settings['export_inpainted_image']:
            path = os.path.join(item.directory, f"comic_translate_{timestamp}", "cleaned_images", item.archive_bname)
            os.makedirs(path, exist_ok=True)
            imk.write_image(os.path.join(path, f"{item.base_name}_cleaned{item.extension}"), inpaint_input_img)

        extra_context = settings_page.get_llm_settings()['extra_context']
        translator_key = settings_page.get_tool_selection('translator')
        translator = Translator(context, source_lang, target_lang)
        translation_cache_key = self.cache_manager._get_translation_cache_key(
            image,
            source_lang,
            target_lang,
            translator_key,
            extra_context,
        )
        try:
            translator.translate(blk_list, image, extra_context)
            self.cache_manager._cache_translation_results(translation_cache_key, blk_list)
        except InsufficientCreditsException:
            raise
        except Exception as error:
            err_msg = self._format_pipeline_error(error, 'translation')
            logger.exception("Translation failed: %s", err_msg)
            result.skipped_stage = "Translator"
            result.skipped_message = err_msg
            result.skip_reason = f"Translator: {err_msg}"
            result.skip_traceback = traceback.format_exc()
            return result

        entire_raw_text = get_raw_text(blk_list)
        entire_translated_text = get_raw_translation(blk_list)
        try:
            raw_text_obj = json.loads(entire_raw_text)
            translated_text_obj = json.loads(entire_translated_text)
            if (not raw_text_obj) or (not translated_text_obj):
                result.skipped_stage = "Translator"
                result.skip_reason = "Translator: empty JSON"
                return result
        except json.JSONDecodeError as error:
            error_message = str(error)
            result.skipped_stage = "Translator"
            result.skipped_message = error_message
            result.skip_reason = f"Translator: JSONDecodeError: {error_message}"
            result.skip_traceback = traceback.format_exc()
            logger.exception(result.skip_reason)
            return result

        if export_settings['export_raw_text']:
            path = os.path.join(item.directory, f"comic_translate_{timestamp}", "raw_texts", item.archive_bname)
            os.makedirs(path, exist_ok=True)
            with open(
                os.path.join(path, f"{item.base_name}_raw.json"),
                'w',
                encoding='UTF-8',
            ) as file:
                file.write(entire_raw_text)

        if export_settings['export_translated_text']:
            path = os.path.join(item.directory, f"comic_translate_{timestamp}", "translated_texts", item.archive_bname)
            os.makedirs(path, exist_ok=True)
            with open(
                os.path.join(path, f"{item.base_name}_translated.json"),
                'w',
                encoding='UTF-8',
            ) as file:
                file.write(entire_translated_text)

        target_lang_en = context.lang_mapping.get(target_lang, target_lang)
        trg_lng_cd = get_language_code(target_lang_en)
        upper_case = render_settings.upper_case
        outline = render_settings.outline
        format_translations(blk_list, trg_lng_cd, upper_case=upper_case)
        get_best_render_area(blk_list, image, inpaint_input_img)

        font = render_settings.font_family
        setting_font_color = QColor(render_settings.color)
        max_font_size = render_settings.max_font_size
        min_font_size = render_settings.min_font_size
        line_spacing = float(render_settings.line_spacing)
        outline_width = float(render_settings.outline_width)
        outline_color = QColor(render_settings.outline_color) if outline else None
        bold = render_settings.bold
        italic = render_settings.italic
        underline = render_settings.underline
        alignment = button_to_alignment[render_settings.alignment_id]
        direction = render_settings.direction

        text_items_state = []
        render_emit_items = []
        for blk in blk_list:
            x1, y1, block_width, block_height = blk.xywh
            translation = blk.translation
            if not translation or len(translation) == 1:
                continue

            vertical = is_vertical_block(blk, trg_lng_cd)
            translation, font_size, rendered_width, rendered_height = pyside_word_wrap(
                translation,
                font,
                block_width,
                block_height,
                line_spacing,
                outline_width,
                bold,
                italic,
                underline,
                alignment,
                direction,
                max_font_size,
                min_font_size,
                vertical,
                return_metrics=True,
            )
            render_emit_items.append(_RenderEmitItem(translation, font_size, blk))
            if is_no_space_lang(trg_lng_cd):
                translation = translation.replace(' ', '')
            font_color = get_smart_text_color(blk.font_color, setting_font_color)
            text_props = TextItemProperties(
                text=translation,
                font_family=font,
                font_size=font_size,
                text_color=font_color,
                alignment=alignment,
                line_spacing=line_spacing,
                outline_color=outline_color,
                outline_width=outline_width,
                bold=bold,
                italic=italic,
                underline=underline,
                position=(x1, y1),
                rotation=blk.angle,
                scale=1.0,
                transform_origin=blk.tr_origin_point,
                width=rendered_width,
                height=rendered_height,
                direction=direction,
                vertical=vertical,
                selection_outlines=[
                    OutlineInfo(0, len(translation), outline_color, outline_width, OutlineType.Full_Document)
                ] if outline else [],
            )
            text_items_state.append(text_props.to_dict())

        result.success = True
        result.blk_list = blk_list
        result.text_items_state = text_items_state
        result.render_emit_items = render_emit_items
        return result

    def _apply_result(self, result: _PageResult, timestamp: str, total_images: int):
        image_path = result.image_path
        if result.skipped_stage:
            self.skip_save(result.directory, timestamp, result.base_name, result.extension, result.archive_bname, None)
            if result.skipped_stage != "User-skipped":
                self.main_page.image_skipped.emit(image_path, result.skipped_stage, result.skipped_message)
            self.log_skipped_image(
                result.directory,
                timestamp,
                image_path,
                result.skip_reason,
                result.skip_traceback,
            )
            return

        state = self.main_page.image_states[image_path]
        state['viewer_state'].update({'text_items_state': result.text_items_state})
        state['viewer_state'].update({'push_to_stack': True})
        state.update({'blk_list': result.blk_list})

        self.main_page.patches_processed.emit(result.patches, image_path)

        current_display = self.main_page.image_files[self.main_page.curr_img_idx]
        if image_path == current_display:
            for item in result.render_emit_items:
                self.main_page.blk_rendered.emit(item.text, item.font_size, item.block, image_path)
            self.main_page.blk_list = result.blk_list

        self.main_page.render_state_ready.emit(image_path)
        self.emit_progress(result.index, total_images, 10, 10, False)

    def batch_process(self, selected_paths: List[str] = None):
        timestamp = datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p")
        image_list = selected_paths if selected_paths is not None else self.main_page.image_files
        total_images = len(image_list)
        try:
            if self.main_page.file_handler.should_pre_materialize(image_list):
                count = self.main_page.file_handler.pre_materialize(image_list)
                logger.info("Batch pre-materialized %d paths before full-run processing.", count)
        except Exception:
            logger.debug("Batch pre-materialization failed; continuing lazily.", exc_info=True)

        settings_snapshot = _SettingsSnapshot.from_settings_page(self.main_page.settings_page)
        context = _BatchPageContext(settings_snapshot, dict(self.main_page.lang_mapping))
        render_settings = self.main_page.render_settings()
        button_to_alignment = dict(self.main_page.button_to_alignment)
        work_items = self._build_work_items(image_list)

        for item in work_items:
            if self._is_cancelled():
                return

            self.emit_progress(item.index, total_images, 0, 10, True)
            result = self._process_single_page(
                item,
                timestamp,
                settings_snapshot,
                context,
                render_settings,
                button_to_alignment,
            )

            if self._is_cancelled():
                return

            self._apply_result(result, timestamp, total_images)
