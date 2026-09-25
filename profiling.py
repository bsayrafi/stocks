"""
profiling.py
------------
Lightweight per-step timing for the screener.

    from profiling import _profiled, profile_step, reset_profile, print_profile_summary

    @_profiled("pre-market (Alpaca batch, all tickers)")
    def get_best_pre_market_prices(tickers, ...): ...

    with profile_step("analyst data", symbol):
        ...

PROFILE maps step name -> list of (ticker or label, seconds). Step names that
start with two spaces are treated as sub-steps (shown, but not added to the
"total work" figure). The special step "whole ticker (run_screen)" holds one
entry per ticker and is used for the "slowest tickers" line.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from functools import wraps

import constants

PROFILE: dict[str, list[tuple[str, float]]] = {}
_PROFILE_LOCK = threading.Lock()


def _record(step: str, label: str, seconds: float) -> None:
    with _PROFILE_LOCK:
        PROFILE.setdefault(step, []).append((label, seconds))


def _label_from_args(args) -> str:
    """Ticker for single-ticker calls, 'N tickers' for batch calls."""
    if not args:
        return "-"
    first = args[0]
    if isinstance(first, str):
        return first
    try:
        return f"{len(first)} tickers"
    except TypeError:
        return str(first)


def _profiled(step: str):
    """Decorator: time every call of the function under `step`."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _record(step, _label_from_args(args), time.perf_counter() - t0)
        return wrapper
    return deco


@contextmanager
def profile_step(step: str, label: str = "-", enabled: bool = True):
    """Context manager: time a block of code under `step`.
    enabled=False runs the block without recording (e.g. a feature switched
    off in CONFIG), so it doesn't show up as a 0-second row in the summary."""
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _record(step, label, time.perf_counter() - t0)


def reset_profile() -> None:
    """Clear all timings (call at the start of each screener run)."""
    with _PROFILE_LOCK:
        PROFILE.clear()


def print_profile_summary(wall_seconds: float, n_tickers: int, max_workers: int | None = None) -> None:
    """Print where the time went, biggest step first."""
    if not PROFILE:
        return
    if max_workers is None:
        max_workers = constants.CONFIG.get("MAX_WORKERS", 1)
    with _PROFILE_LOCK:
        snapshot = {k: list(v) for k, v in PROFILE.items()}

    rows = []
    for step, calls in snapshot.items():
        total = sum(sec for _, sec in calls)
        slow_t, slow_s = max(calls, key=lambda c: c[1])
        rows.append((step, total, len(calls), total / len(calls), slow_t, slow_s))
    rows.sort(key=lambda r: -r[1])
    work = sum(r[1] for r in rows if not r[0].startswith("  ") and r[0] != "whole ticker (run_screen)")

    print(f"\n=== Timing summary: {n_tickers} tickers, {wall_seconds:.1f}s wall time "
          f"({wall_seconds / max(n_tickers, 1):.1f}s per ticker) ===")
    print(f"total work {work:.1f}s across {max_workers} parallel workers "
          f"(steps overlap, so work > wall time)")
    print(f"{'step':<40}{'total s':>9}{'% work':>8}{'calls':>7}{'avg s':>8}   slowest")
    for step, total, n, avg, slow_t, slow_s in rows:
        if step == "whole ticker (run_screen)":
            continue
        print(f"{step:<40}{total:>9.1f}{total / max(work, 1e-9) * 100:>7.1f}%{n:>7}{avg:>8.2f}"
              f"   {slow_t} {slow_s:.1f}s")
    per_ticker = sorted(snapshot.get("whole ticker (run_screen)", []), key=lambda c: -c[1])[:5]
    if per_ticker:
        print("slowest tickers: " + ", ".join(f"{t} {sec:.1f}s" for t, sec in per_ticker))
    print("(sub-steps marked '  analyst:' are included in 'analyst data')\n")
