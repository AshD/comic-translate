import os
import logging
import shutil
import tempfile
import threading

from .archives import (
    list_archive_image_entries,
    materialize_archive_entry,
    materialize_archive_entries,
)

_LAZY_SOURCE_LOCK = threading.RLock()
_LAZY_SOURCE_BY_PATH: dict[str, dict] = {}
logger = logging.getLogger(__name__)


def _register_lazy_source(path: str, source: dict) -> None:
    with _LAZY_SOURCE_LOCK:
        _LAZY_SOURCE_BY_PATH[os.path.abspath(path)] = source


def _clear_lazy_sources_under_dir(base_dir: str) -> None:
    base = os.path.abspath(base_dir)
    with _LAZY_SOURCE_LOCK:
        stale_paths = [p for p in _LAZY_SOURCE_BY_PATH if p.startswith(base)]
        for p in stale_paths:
            _LAZY_SOURCE_BY_PATH.pop(p, None)


def ensure_prepared_path_materialized(path: str) -> bool:
    if not path:
        return False
    abs_path = os.path.abspath(path)
    try:
        if os.path.isfile(abs_path) and os.path.getsize(abs_path) > 0:
            logger.debug("Prepared path already materialized: %s", abs_path)
            return True
    except Exception:
        pass

    with _LAZY_SOURCE_LOCK:
        source = _LAZY_SOURCE_BY_PATH.get(abs_path)
    if source is None:
        exists = os.path.isfile(abs_path)
        logger.debug("No lazy source for path=%s exists=%s", abs_path, exists)
        return exists

    archive_path = str(source.get("archive_path", ""))
    entry = source.get("entry")
    if not archive_path or not isinstance(entry, dict):
        logger.debug("Invalid lazy source for path=%s archive=%s", abs_path, archive_path)
        return False

    logger.info("Materializing page path=%s from archive=%s entry=%s", abs_path, archive_path, entry)
    ok = materialize_archive_entry(archive_path, entry, abs_path)
    logger.info("Materialized page path=%s success=%s", abs_path, ok)
    return ok


class FileHandler:
    def __init__(self):
        self.file_paths = []
        self.archive_info = []

    def _find_archive_for_path(self, path: str):
        abs_path = os.path.abspath(path)
        for archive in self.archive_info:
            extracted_images = archive.get('extracted_images', [])
            if abs_path in {os.path.abspath(p) for p in extracted_images}:
                return archive
        return None

    def prepare_files(self, file_paths: list[str], extend: bool = False):
        all_image_paths = []
        if not extend:
            for archive in self.archive_info:
                temp_dir = archive['temp_dir']
                _clear_lazy_sources_under_dir(temp_dir)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            self.archive_info = []

        for path in file_paths:
            if path.lower().endswith((
                '.cbr', '.cbz', '.cbt', '.cb7',
                '.zip', '.rar', '.7z', '.tar',
                '.pdf', '.epub',
            )):
                print('Indexing archive:', path)
                archive_dir = os.path.dirname(path)
                temp_dir = tempfile.mkdtemp(dir=archive_dir)

                entries = list_archive_image_entries(path)
                total = len(entries)
                digits = len(str(total)) if total > 0 else 1
                image_paths: list[str] = []

                for index, entry in enumerate(entries, start=1):
                    ext = str(entry.get("ext", ".png"))
                    if not ext.startswith("."):
                        ext = f".{ext}"
                    lazy_path = os.path.join(temp_dir, f"{index:0{digits}d}{ext.lower()}")
                    _register_lazy_source(
                        lazy_path,
                        {"archive_path": path, "entry": entry},
                    )
                    image_paths.append(lazy_path)

                # Improve first paint latency by ensuring page 1 is ready.
                if image_paths:
                    logger.info("Preparing first page for archive=%s", path)
                    ensure_prepared_path_materialized(image_paths[0])
                    logger.info("First page ready for archive=%s", path)

                all_image_paths.extend(image_paths)
                self.archive_info.append({
                    'archive_path': path,
                    'extracted_images': image_paths,
                    'temp_dir': temp_dir,
                })
            else:
                all_image_paths.append(path)

        self.file_paths = self.file_paths + all_image_paths if extend else all_image_paths
        return all_image_paths

    def should_pre_materialize(self, target_paths: list[str] | None = None) -> bool:
        paths = list(target_paths or [])
        if not paths:
            return False
        all_paths = list(self.file_paths or [])
        if not all_paths:
            return False

        # High-DPI PDF rasterization is expensive. Materialize those lazily so
        # batch progress starts immediately instead of spending minutes at 0%.
        for path in paths:
            archive = self._find_archive_for_path(path)
            if archive and str(archive.get('archive_path', '')).lower().endswith('.pdf'):
                return False

        target_count = len(set(paths))
        total_count = max(1, len(set(all_paths)))
        ratio = target_count / total_count
        return target_count == total_count or ratio >= 0.7

    def pre_materialize(self, target_paths: list[str] | None = None) -> int:
        paths = list(target_paths or self.file_paths or [])
        if not paths:
            return 0

        grouped: dict[str, list[tuple[dict, str]]] = {}
        fallback_paths: list[str] = []

        for path in paths:
            abs_path = os.path.abspath(path)
            try:
                if os.path.isfile(abs_path) and os.path.getsize(abs_path) > 0:
                    continue
            except Exception:
                pass

            with _LAZY_SOURCE_LOCK:
                source = _LAZY_SOURCE_BY_PATH.get(abs_path)
            if source is None:
                continue

            archive_path = str(source.get("archive_path", ""))
            entry = source.get("entry")
            if archive_path and isinstance(entry, dict):
                grouped.setdefault(archive_path, []).append((entry, abs_path))
            else:
                fallback_paths.append(abs_path)

        completed = 0
        for archive_path, items in grouped.items():
            completed += materialize_archive_entries(archive_path, items)

        for abs_path in fallback_paths:
            with _LAZY_SOURCE_LOCK:
                source = _LAZY_SOURCE_BY_PATH.get(abs_path)
            if source is None:
                continue
            archive_path = str(source.get("archive_path", ""))
            entry = source.get("entry")
            if archive_path and isinstance(entry, dict):
                if materialize_archive_entry(archive_path, entry, abs_path):
                    completed += 1

        return completed
