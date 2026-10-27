"""Lazy, cached model loading with dynamic CUDA/CPU allocation.

A failed load is remembered so a missing model never slows down every request;
callers receive None and use their heuristic fallback.
"""
import threading

from flask import current_app

from .logger import get_logger

log = get_logger(__name__)
_cache = {}
_failed = {}
_lock = threading.Lock()


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_model(key, loader):
    """Return loader() result cached under `key`, or None if unavailable."""
    if not current_app.config.get("USE_ML_MODELS", True):
        return None
    with _lock:
        if key in _cache:
            return _cache[key]
        if key in _failed:
            return None
        try:
            log.info("Loading model '%s' on %s ...", key, get_device())
            _cache[key] = loader()
            log.info("Model '%s' ready", key)
            return _cache[key]
        except Exception as exc:  # missing package, offline, out of memory ...
            _failed[key] = str(exc)
            log.warning("Model '%s' unavailable (%s). Using fallback.", key, exc)
            return None


def status() -> dict:
    return {
        "device": get_device(),
        "loaded": sorted(_cache),
        "failed": dict(_failed),
    }


def reset():
    with _lock:
        _cache.clear()
        _failed.clear()
