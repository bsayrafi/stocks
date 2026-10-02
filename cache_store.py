"""
cache_store.py - ONE disk cache for the screener scripts (tickersV2.py, enrich_html_v2.py, ...).

Where the cache lives, first match wins:
  1. environment variable CACHE_DIR, or the older FINVIZ_CACHE_DIR (e.g. set by a GitHub Actions workflow)
  2. configure(dir=...) from a script, e.g. enrich_html_v2's CONFIGH["CACHE_DIR"] when it is set
  3. constants.CONFIG["CACHE_DIR"], or the older CONFIG["FINVIZ_CACHE_DIR"]
  4. a folder "htmlv2cache" next to this file
A relative path from 2-4 is taken relative to THIS file's folder (so it does not depend on where Python was started);
a relative path in an environment variable is taken relative to the current working directory, as before.

Every file name starts with today's date (YYYYMMDD_<name>_<key>.pkl). prune() deletes files from earlier days.

    from cache_store import daily_cached
    info = daily_cached("info", "AAPL", lambda: fetch_info("AAPL"))      # network at most once a day per ticker
"""
import datetime as dt
import os
import pickle
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_lock = threading.Lock()
_settings = {"dir": None, "enabled": True, "hook": None}
_write_warned = False


def configure(dir=None, enabled=None, hook=None) -> None:
    """dir: cache folder (lower priority than the CACHE_DIR / FINVIZ_CACHE_DIR environment variables).
    enabled: False turns caching off (every call goes to the network).
    hook: hook(event, name, key) with event "hit" or "miss" - e.g. for a timing summary."""
    if dir is not None:
        _settings["dir"] = dir
    if enabled is not None:
        _settings["enabled"] = bool(enabled)
    if hook is not None:
        _settings["hook"] = hook


def enabled() -> bool:
    return _settings["enabled"]


def cache_dir() -> str:
    env = os.environ.get("CACHE_DIR") or os.environ.get("FINVIZ_CACHE_DIR")
    if env:
        return os.path.abspath(env)
    d = _settings["dir"]
    if not d:
        try:
            from constants import CONFIG
            d = CONFIG.get("CACHE_DIR") or CONFIG.get("FINVIZ_CACHE_DIR")
        except Exception:
            d = None
    d = d or "htmlv2cache"
    return d if os.path.isabs(d) else os.path.join(_HERE, d)


def cache_path(name: str, key: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(key))
    return os.path.join(cache_dir(), f"{dt.date.today():%Y%m%d}_{name}_{safe}.pkl")


def _event(event: str, name: str, key: str) -> None:
    h = _settings["hook"]
    if h is not None:
        try:
            h(event, name, key)
        except Exception:
            pass


def write(path: str, obj) -> None:
    """Atomic write (temp file + rename, so parallel workers never read half a file).
    A failure is printed once per run instead of being silently ignored."""
    global _write_warned
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)
    except Exception as e:
        with _lock:
            if not _write_warned:
                _write_warned = True
                print(f"WARNING: cache write failed ({type(e).__name__}: {e}) - nothing will be cached this run. Path: {path}")


def read(path: str, default=None):
    """The pickled object at path, or default if it is missing or unreadable."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        return default
    except Exception as e:                         # e.g. written by another library version: treat as missing
        print(f"  cache file unreadable ({type(e).__name__}) - refetching: {os.path.basename(path)}")
        return default


_MISSING = object()


def daily_cached(name: str, key: str, fn, is_valid=bool, refresh: bool = False):
    """Today's cached value for (name, key), or fn() - cached when is_valid(result). Failures are never cached.
    refresh=True ignores today's file and fetches again (the new result replaces it)."""
    if not _settings["enabled"]:
        return fn()
    path = cache_path(name, key)
    if not refresh:
        value = read(path, _MISSING)
        if value is not _MISSING:
            _event("hit", name, key)
            return value
    _event("miss", name, key)
    value = fn()
    if is_valid(value):
        write(path, value)
    return value


def ttl_cached(name: str, key: str, fn, ttl_minutes: float, is_valid=lambda v: v is not None):
    """Like daily_cached, but a value is only reused while it is younger than ttl_minutes (e.g. news: 60).
    The fetch time is stored INSIDE the file, so restoring the folder (e.g. a GitHub Actions cache) keeps ages right."""
    if not _settings["enabled"] or not ttl_minutes:
        return fn()
    path = cache_path(name, key)
    got = read(path)
    if isinstance(got, tuple) and len(got) == 2 and time.time() - got[0] < ttl_minutes * 60:
        _event("hit", name, key)
        return got[1]
    _event("miss", name, key)
    value = fn()
    if is_valid(value):
        write(path, (time.time(), value))
    return value


def prune(verbose: bool = True) -> None:
    """Delete cache files from earlier days and (verbose) print where the cache is and what today's run will find."""
    d = cache_dir()
    today = f"{dt.date.today():%Y%m%d}_"
    n_today = n_old = 0
    try:
        for fn in os.listdir(d):
            if not fn.endswith(".pkl"):
                continue
            if fn.startswith(today):
                n_today += 1
                continue
            try:
                os.remove(os.path.join(d, fn))
                n_old += 1
            except OSError:
                pass
    except OSError:
        pass
    if verbose:
        if _settings["enabled"]:
            print(f"cache: {d} - {n_today} file(s) from today" + (f", {n_old} old file(s) deleted" if n_old else "")
                  + (" (first run today: everything is fetched)" if n_today == 0 else ""))
        else:
            print("cache: OFF")
