import hashlib
import logging
import threading

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages OCR and translation caching for the pipeline."""

    def __init__(self):
        self.ocr_cache = {}
        self.translation_cache = {}
        self._lock = threading.RLock()

    def clear_ocr_cache(self):
        """Clear the OCR cache. Note: Cache now persists across image and model changes automatically."""
        with self._lock:
            self.ocr_cache = {}
        logger.info("OCR cache manually cleared")

    def clear_translation_cache(self):
        """Clear the translation cache. Note: Cache now persists across image and model changes automatically."""
        with self._lock:
            self.translation_cache = {}
        logger.info("Translation cache manually cleared")

    def _generate_image_hash(self, image):
        """Generate a hash for the image to use as cache key."""
        try:
            sample_data = image[::10, ::10].tobytes()
            return hashlib.md5(sample_data).hexdigest()
        except Exception:
            shape_str = str(image.shape) if hasattr(image, 'shape') else str(type(image))
            fallback_data = shape_str.encode() + str(image.dtype).encode() if hasattr(image, 'dtype') else b'fallback'
            return hashlib.md5(fallback_data).hexdigest()

    def _get_ocr_cache_key(self, image, source_lang, ocr_model, device=None):
        image_hash = self._generate_image_hash(image)
        if device is None:
            device = 'unknown'
        return (image_hash, ocr_model, source_lang, device)

    def _get_block_id(self, block):
        try:
            x1, y1, x2, y2 = block.xyxy
            angle = block.angle
            return f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}_{int(angle)}"
        except (AttributeError, ValueError, TypeError):
            return str(id(block))

    def _find_matching_block_id(self, cache_key, target_block):
        target_id = self._get_block_id(target_block)
        with self._lock:
            cached_results = dict(self.ocr_cache.get(cache_key, {}))

        if target_id in cached_results:
            return target_id, cached_results[target_id]

        try:
            target_x1, target_y1, target_x2, target_y2 = target_block.xyxy
            target_angle = getattr(target_block, 'angle', 0)
            tolerance = 5.0
            for cached_id in cached_results.keys():
                try:
                    parts = cached_id.split('_')
                    if len(parts) < 5:
                        continue
                    cached_x1 = float(parts[0])
                    cached_y1 = float(parts[1])
                    cached_x2 = float(parts[2])
                    cached_y2 = float(parts[3])
                    cached_angle = float(parts[4])
                    if (
                        abs(target_x1 - cached_x1) <= tolerance
                        and abs(target_y1 - cached_y1) <= tolerance
                        and abs(target_x2 - cached_x2) <= tolerance
                        and abs(target_y2 - cached_y2) <= tolerance
                        and abs(target_angle - cached_angle) <= 1.0
                    ):
                        logger.debug("Fuzzy match found for OCR: %s... -> %s...", target_id[:20], cached_id[:20])
                        return cached_id, cached_results[cached_id]
                except (ValueError, IndexError):
                    continue
        except (AttributeError, ValueError, TypeError):
            pass
        return None, ""

    def _find_matching_translation_block_id(self, cache_key, target_block):
        target_id = self._get_block_id(target_block)
        with self._lock:
            cached_results = dict(self.translation_cache.get(cache_key, {}))

        if target_id in cached_results:
            return target_id, cached_results[target_id]

        try:
            target_x1, target_y1, target_x2, target_y2 = target_block.xyxy
            target_angle = getattr(target_block, 'angle', 0)
            tolerance = 5.0
            for cached_id in cached_results.keys():
                try:
                    parts = cached_id.split('_')
                    if len(parts) < 5:
                        continue
                    cached_x1 = float(parts[0])
                    cached_y1 = float(parts[1])
                    cached_x2 = float(parts[2])
                    cached_y2 = float(parts[3])
                    cached_angle = float(parts[4])
                    if (
                        abs(target_x1 - cached_x1) <= tolerance
                        and abs(target_y1 - cached_y1) <= tolerance
                        and abs(target_x2 - cached_x2) <= tolerance
                        and abs(target_y2 - cached_y2) <= tolerance
                        and abs(target_angle - cached_angle) <= 1.0
                    ):
                        logger.debug("Fuzzy match found for translation: %s... -> %s...", target_id[:20], cached_id[:20])
                        return cached_id, cached_results[cached_id]
                except (ValueError, IndexError):
                    continue
        except (AttributeError, ValueError, TypeError):
            pass
        return None, ""

    def _is_ocr_cached(self, cache_key):
        with self._lock:
            return cache_key in self.ocr_cache

    def _cache_ocr_results(self, cache_key, blk_list, processed_blk_list=None):
        try:
            block_results = {}
            if processed_blk_list is not None:
                for original_blk, processed_blk in zip(blk_list, processed_blk_list):
                    block_id = self._get_block_id(original_blk)
                    text = getattr(processed_blk, 'text', '') or ''
                    if text:
                        block_results[block_id] = text
            else:
                for blk in blk_list:
                    block_id = self._get_block_id(blk)
                    text = getattr(blk, 'text', '') or ''
                    if text:
                        block_results[block_id] = text
            if block_results:
                with self._lock:
                    self.ocr_cache[cache_key] = block_results
                logger.info("Cached OCR results for %d blocks", len(block_results))
            else:
                logger.debug("No OCR text found in blocks; skipping OCR cache creation")
        except Exception as error:
            logger.warning("Failed to cache OCR results: %s", error)

    def update_ocr_cache_for_block(self, cache_key, block):
        block_id = self._get_block_id(block)
        text = getattr(block, 'text', '') or ''
        if not text:
            logger.debug("Skipping OCR cache update for empty text for block ID %s", block_id)
            return
        with self._lock:
            if cache_key not in self.ocr_cache:
                self.ocr_cache[cache_key] = {}
            self.ocr_cache[cache_key][block_id] = text
        logger.debug("Updated OCR cache for block ID %s", block_id)

    def _get_cached_text_for_block(self, cache_key, block):
        matched_id, result = self._find_matching_block_id(cache_key, block)
        if matched_id is not None:
            return result
        block_id = self._get_block_id(block)
        with self._lock:
            cached_results = dict(self.ocr_cache.get(cache_key, {}))
        logger.debug("No cached text found for block ID %s", block_id)
        logger.debug("Available block IDs in cache: %s", list(cached_results.keys()))
        return None

    def _get_translation_cache_key(self, image, source_lang, target_lang, translator_key, extra_context):
        image_hash = self._generate_image_hash(image)
        context_hash = hashlib.md5(extra_context.encode()).hexdigest() if extra_context else 'no_context'
        return (image_hash, translator_key, source_lang, target_lang, context_hash)

    def _is_translation_cached(self, cache_key):
        with self._lock:
            return cache_key in self.translation_cache

    def _cache_translation_results(self, cache_key, blk_list, processed_blk_list=None):
        try:
            block_results = {}
            if processed_blk_list is not None:
                for original_blk, processed_blk in zip(blk_list, processed_blk_list):
                    block_id = self._get_block_id(original_blk)
                    translation = getattr(processed_blk, 'translation', '') or ''
                    source_text = getattr(original_blk, 'text', '') or ''
                    if translation:
                        block_results[block_id] = {
                            'source_text': source_text,
                            'translation': translation,
                        }
            else:
                for blk in blk_list:
                    block_id = self._get_block_id(blk)
                    translation = getattr(blk, 'translation', '') or ''
                    source_text = getattr(blk, 'text', '') or ''
                    if translation:
                        block_results[block_id] = {
                            'source_text': source_text,
                            'translation': translation,
                        }
            if block_results:
                with self._lock:
                    self.translation_cache[cache_key] = block_results
                logger.info("Cached translation results for %d blocks", len(block_results))
            else:
                logger.debug("No translations found in blocks; skipping translation cache creation")
        except Exception as error:
            logger.warning("Failed to cache translation results: %s", error)

    def update_translation_cache_for_block(self, cache_key, block):
        block_id = self._get_block_id(block)
        translation = getattr(block, 'translation', '') or ''
        source_text = getattr(block, 'text', '') or ''
        if not translation:
            logger.debug("Skipping translation cache update for empty translation for block ID %s", block_id)
            return
        with self._lock:
            if cache_key not in self.translation_cache:
                self.translation_cache[cache_key] = {}
            self.translation_cache[cache_key][block_id] = {
                'source_text': source_text,
                'translation': translation,
            }
        logger.debug("Updated translation cache for block ID %s", block_id)

    def _get_cached_translation_for_block(self, cache_key, block):
        matched_id, result = self._find_matching_translation_block_id(cache_key, block)
        if matched_id is not None:
            if result:
                cached_source_text = result.get('source_text', '')
                current_source_text = getattr(block, 'text', '') or ''
                if cached_source_text == current_source_text:
                    return result.get('translation', '')
                logger.debug(
                    "Cache invalid: source text changed from '%s' to '%s'",
                    cached_source_text,
                    current_source_text,
                )
                return None
            return ''
        block_id = self._get_block_id(block)
        with self._lock:
            cached_results = dict(self.translation_cache.get(cache_key, {}))
        logger.debug("No cached translation found for block ID %s", block_id)
        logger.debug("Available block IDs in cache: %s", list(cached_results.keys()))
        return None

    def _can_serve_all_blocks_from_ocr_cache(self, cache_key, block_list):
        if not self._is_ocr_cached(cache_key):
            return False
        for block in block_list:
            if self._get_cached_text_for_block(cache_key, block) is None:
                return False
        return True

    def _can_serve_all_blocks_from_translation_cache(self, cache_key, block_list):
        if not self._is_translation_cached(cache_key):
            return False
        for block in block_list:
            if self._get_cached_translation_for_block(cache_key, block) is None:
                return False
        return True

    def _apply_cached_ocr_to_blocks(self, cache_key, block_list):
        for block in block_list:
            cached_text = self._get_cached_text_for_block(cache_key, block)
            if cached_text is not None:
                block.text = cached_text

    def _apply_cached_translations_to_blocks(self, cache_key, block_list):
        for block in block_list:
            cached_translation = self._get_cached_translation_for_block(cache_key, block)
            if cached_translation is not None:
                block.translation = cached_translation
