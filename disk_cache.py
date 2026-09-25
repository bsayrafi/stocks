"""
disk_cache.py
-------------
Small on-disk cache for slow, slowly-changing per-ticker data (earnings dates,
cash flow statements). One pickle file per namespace under CACHE_DIR, loaded
lazily, updated in memory (thread-safe), and written back by flush_all().

On GitHub Actions the CACHE_DIR folder is carried between runs by the
actions/cache step in the workflow file.

    from disk_cache import cached, flush_all

    df = cached("earnings_dates", "AAPL", ttl_days=7, fetch=lambda: ...)
    ...
    flush_all()   # once, after the parallel loop
"""

from __future__ import annotations

import atexit
import os
import pickle
import threading
import time

import constants

CACHE_DIR = os.path.join(constants.WORK_DIR, "cache")

_STORES: dict[str, dict] = {}      # namespace -> {key: (saved_at_epoch, value)}
_DIRTY: set[str] = set()
_LOCK = threading.Lock()
_STATS: dict[str, list[int]] = {}  # namespace -> [hits, misses]


def _path(namespace: str) -> str:
    return os.path.join(CACHE_DIR, f"{namespace}.pkl")


def _load(namespace: str) -> dict:
    """Load a namespace from disk once (caller holds _LOCK)."""
    store = _STORES.get(namespace)
    if store is None:
        store = {}
        try:
            with open(_path(namespace), "rb") as f:
                store = pickle.load(f)
        except FileNotFoundError:
            pass
        except Exception as e:  # corrupt / incompatible file -> start fresh
            print(f"  cache '{namespace}' unreadable ({e}); starting empty")
        _STORES[namespace] = store
    return store


def get(namespace: str, key: str, ttl_days: float):
    """Cached value if present and younger than ttl_days, else None."""
    with _LOCK:
        entry = _load(namespace).get(key)
    if entry is None:
        return None
    saved_at, value = entry
    if time.time() - saved_at > ttl_days * 86400:
        return None
    return value


def put(namespace: str, key: str, value) -> None:
    with _LOCK:
        _load(namespace)[key] = (time.time(), value)
        _DIRTY.add(namespace)


def record(namespace: str, hit: bool) -> None:
    """Count a cache hit or miss (printed by flush_all)."""
    with _LOCK:
        _STATS.setdefault(namespace, [0, 0])[0 if hit else 1] += 1


def cached(namespace: str, key: str, ttl_days: float, fetch):
    """Return the cached value, or call fetch(), cache a non-None result, and
    return it. Failures (None) are not cached, so they are retried next run."""
    value = get(namespace, key, ttl_days)
    record(namespace, value is not None)
    if value is not None:
        return value
    value = fetch()
    if value is not None:
        put(namespace, key, value)
    return value


def flush_all() -> None:
    """Write every changed namespace back to disk (atomic replace)."""
    with _LOCK:
        dirty = {ns: dict(_STORES[ns]) for ns in _DIRTY}
        _DIRTY.clear()
        stats = {ns: list(v) for ns, v in _STATS.items()}
        _STATS.clear()
    if not dirty and not stats:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    for ns, store in dirty.items():
        tmp = _path(ns) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, _path(ns))
    for ns, (hits, misses) in sorted(stats.items()):
        print(f"  cache '{ns}': {hits} hit(s), {misses} fetched")


atexit.register(flush_all)
