"""
event_catalysts.py

Event-driven / catalyst-based signals for a single ticker.
Complements a technical/fundamental scorecard by flagging things that
move price on their own schedule, independent of chart patterns:

    - earnings date proximity (and the historical vol spike around it)
    - S&P 500 index membership (proxy for inclusion/exclusion flow)
    - buyback announcements (via recent news headlines)
    - guidance changes (via recent news headlines)
    - analyst upgrade/downgrade clustering (via recent rating actions)

Data source: yfinance (Yahoo Finance). Free, no API key, but it means
the numbers are only as reliable/fresh as Yahoo's feed -- treat this as
a screening signal, not a source of truth for trade timing.

Install: pip install yfinance pandas
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field

import pandas as pd
import yfinance as yf
import requests

_APEWISDOM_EMPTY = {
    "mentions": 0,
    "upvotes": 0,
    "rank": "N/A",
    "mentions_24h_ago": 0,
    "rank_24h_ago": "N/A",
    "momentum_pct": "N/A",
}


def fetch_apewisdom_table(pages: int = 1) -> dict[str, dict] | None:
    """Download ApeWisdom's ranking ONCE (100 stocks per page) and return
    {TICKER: sentiment dict}. Pass the result to get_event_catalysts(social_table=...)
    so a watchlist run makes 1 request instead of 1 per ticker.
    Returns None if the download fails."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; catalyst-screener/1.0)"}
    table: dict[str, dict] = {}
    try:
        for page in range(1, pages + 1):
            url = f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                break
            for item in resp.json().get("results", []):
                mentions = int(item.get("mentions", 0))
                mentions_24h_ago = int(item.get("mentions_24h_ago", 0) or 0)
                momentum_pct = (round((mentions - mentions_24h_ago) / mentions_24h_ago * 100, 1)
                                if mentions_24h_ago > 0 else "N/A")
                table[str(item.get("ticker", "")).upper()] = {
                    "mentions": mentions,
                    "upvotes": int(item.get("upvotes", 0)),
                    "rank": int(item.get("rank", 999)),
                    "mentions_24h_ago": mentions_24h_ago,
                    "rank_24h_ago": int(item.get("rank_24h_ago", 999) or 999),
                    "momentum_pct": momentum_pct,
                }
    except Exception:
        return table or None
    return table


def get_apewisdom_sentiment(ticker: str, table: dict[str, dict] | None = None) -> dict:
    """Reddit mention volume, upvotes, and 24h momentum from ApeWisdom.
    Looks the ticker up in a pre-fetched `table` (see fetch_apewisdom_table);
    without one, downloads the ranking for this single call."""
    if table is None:
        table = fetch_apewisdom_table() or {}
    return dict(table.get(ticker.upper(), _APEWISDOM_EMPTY))

@dataclass
class CatalystReport:
    ticker: str
    as_of: dt.date

    social_mentions: int = 0
    social_upvotes: int = 0
    social_rank: int | str = "N/A"
    social_mentions_24h_ago: int = 0
    social_rank_24h_ago: int | str = "N/A"
    social_momentum_pct: float | str = "N/A"
    # Earnings
    next_earnings_date: dt.date | None = None
    days_to_earnings: int | None = None
    in_earnings_window: bool = False  # within +/- 3 trading days

    # Index membership
    in_sp500: bool | None = None  # None = couldn't determine
    sp500_added_date: dt.date | None = None
    days_since_index_addition: int | None = None
    recent_index_addition: bool = False  # within ~60 calendar days

    # News-derived catalysts (last N days)
    buyback_headlines: list[str] = field(default_factory=list)
    guidance_headlines: list[str] = field(default_factory=list)

    # Analyst actions (last N days)
    upgrades: int = 0
    downgrades: int = 0
    rating_actions: list[str] = field(default_factory=list)


    
    def summary(self) -> str:
        lines = [f"Catalyst report for {self.ticker} (as of {self.as_of}):"]

        if self.next_earnings_date:
            flag = " <-- inside +/-3 day window" if self.in_earnings_window else ""
            lines.append(
                f"  Earnings: {self.next_earnings_date} "
                f"({self.days_to_earnings:+d} trading days){flag}"
            )
        else:
            lines.append("  Earnings: no confirmed date found")

        if self.in_sp500 is None:
            lines.append("  S&P 500 membership: unknown (lookup failed)")
        elif self.in_sp500:
            extra = ""
            if self.sp500_added_date:
                extra = f" (added {self.sp500_added_date}, {self.days_since_index_addition}d ago)"
                if self.recent_index_addition:
                    extra += " <-- recent addition, possible flow effect"
            lines.append(f"  S&P 500 member: True{extra}")
        else:
            lines.append("  S&P 500 member: False")

        lines.append(f"  Buyback headlines (recent): {len(self.buyback_headlines)}")
        for h in self.buyback_headlines[:3]:
            lines.append(f"    - {h}")

        lines.append(f"  Guidance headlines (recent): {len(self.guidance_headlines)}")
        for h in self.guidance_headlines[:3]:
            lines.append(f"    - {h}")

        lines.append(
            f"  Analyst actions (recent): {self.upgrades} upgrades, "
            f"{self.downgrades} downgrades"
        )
        for a in self.rating_actions[:5]:
            lines.append(f"    - {a}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Flat, JSON-serializable representation (dates -> ISO strings)."""
        d = asdict(self)
        for k in ("as_of", "next_earnings_date", "sp500_added_date"):
            if d.get(k) is not None:
                d[k] = d[k].isoformat()
        return d


def _get_sp500_membership(verbose: bool = False) -> dict[str, dt.date] | None:
    """
    Return {ticker: date_added_to_index} for current S&P 500 constituents.

    Primary source: a GitHub-hosted CSV (github.com/datasets/s-and-p-500-companies),
    which is more reliable to fetch than scraping Wikipedia directly (Wikipedia
    often 403s a bare urllib/pandas request with no browser User-Agent, which is
    the most common reason this lookup silently fails).

    Falls back to scraping Wikipedia's table if the GitHub source is unreachable.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; catalyst-report/1.0)"}

    # --- Primary: GitHub CSV (has a "Date added" column, which Wikipedia's
    #     table doesn't reliably expose in a clean, parseable format) --------
    try:
        import requests

        url = (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/master/data/constituents.csv"
        )
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        result = {}
        for _, row in df.iterrows():
            sym = str(row[symbol_col]).replace(".", "-")
            added_raw = row.get("Date added")
            added_date = None
            if pd.notna(added_raw):
                try:
                    added_date = pd.to_datetime(added_raw).date()
                except Exception:
                    added_date = None
            result[sym] = added_date
        if result:
            return result
    except Exception as e:
        if verbose:
            print(f"  [sp500 lookup] GitHub CSV source failed: {e}")

    # --- Fallback: scrape Wikipedia with an explicit User-Agent -------------
    try:
        import requests

        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        tables = pd.read_html(pd.io.common.StringIO(resp.text))
        df = tables[0]
        result = {}
        for _, row in df.iterrows():
            sym = str(row["Symbol"]).replace(".", "-")
            added_date = None
            if "Date added" in df.columns and pd.notna(row["Date added"]):
                try:
                    added_date = pd.to_datetime(row["Date added"]).date()
                except Exception:
                    added_date = None
            result[sym] = added_date
        return result if result else None
    except Exception as e:
        if verbose:
            print(f"  [sp500 lookup] Wikipedia fallback failed: {e}")
        return None


def get_earnings_and_ratings(
    ticker: str,
    yf_ticker: "yf.Ticker | None" = None,
    info: dict | None = None,
    ratings_lookback_days: int = 30,
) -> dict:
    """The slow, once-a-day part of the catalyst report:
      - next earnings date: taken from `info` (yfinance .info, usually already
        fetched for fundamentals) when it has earnings timestamps; only if it
        has none at all does it fall back to scraping get_earnings_dates()
        (one of yfinance's slowest calls).
      - analyst upgrade/downgrade actions within ratings_lookback_days.
    Returns plain, picklable data so the caller can cache it for the day:
      {"next_earnings_date": date | None, "earnings_source": str,
       "rating_rows": [(date, firm, action, to_grade), ...]}
    """
    today = dt.date.today()
    t = yf_ticker if yf_ticker is not None else yf.Ticker(ticker)
    out = {"next_earnings_date": None, "earnings_source": "none", "rating_rows": []}

    # --- Earnings date: .info first ------------------------------------------
    stamps = [info.get(k) for k in ("earningsTimestampStart", "earningsTimestamp", "earningsTimestampEnd")] \
        if info else []
    stamps = [s for s in stamps if isinstance(s, (int, float)) and s > 0]
    if stamps:
        dates = sorted({dt.datetime.fromtimestamp(s, dt.timezone.utc).date() for s in stamps})
        future = [d for d in dates if d >= today]
        if future:
            out["next_earnings_date"], out["earnings_source"] = future[0], "info"
        else:
            out["earnings_source"] = "info (no upcoming date)"
    else:
        # --- Fallback: scrape the earnings calendar ---------------------------
        try:
            cal = t.get_earnings_dates(limit=8)
            if cal is not None and not cal.empty:
                cal_dates = [d.date() for d in cal.index if hasattr(d, "date")]
                future = [d for d in cal_dates if d >= today]
                if future:
                    out["next_earnings_date"], out["earnings_source"] = min(future), "earnings calendar"
        except Exception:
            pass

    # --- Analyst upgrade/downgrade actions -----------------------------------
    try:
        actions = t.upgrades_downgrades
        if actions is not None and not actions.empty:
            actions = actions.reset_index()
            date_col = "GradeDate" if "GradeDate" in actions.columns else actions.columns[0]
            actions[date_col] = pd.to_datetime(actions[date_col]).dt.date
            recent = actions[actions[date_col] >= today - dt.timedelta(days=ratings_lookback_days)]
            for _, row in recent.iterrows():
                out["rating_rows"].append((row[date_col], str(row.get("Firm", "")),
                                           str(row.get("Action", "")), str(row.get("ToGrade", ""))))
    except Exception:
        pass
    return out


def get_event_catalysts(
    ticker: str,
    news_lookback_days: int = 14,
    ratings_lookback_days: int = 30,
    sp500_members: dict[str, dt.date] | None = None,
    verbose: bool = False,
    *,
    yf_ticker: "yf.Ticker | None" = None,
    info: dict | None = None,
    headlines: list | None = None,
    social_table: dict[str, dict] | None = None,
    earnings_ratings: dict | None = None,
) -> CatalystReport:
    """
    Build a catalyst report for `ticker`.

    Parameters
    ----------
    ticker : str
        e.g. "NVDA"
    news_lookback_days : int
        How far back to scan headlines for buyback/guidance keywords.
    ratings_lookback_days : int
        How far back to scan analyst upgrade/downgrade actions.
    sp500_members : dict[str, date], optional
        Pass a pre-fetched {ticker: date_added} map if calling this in a
        loop, so you don't re-fetch the constituent list for every ticker.
        Get one via `_get_sp500_membership()`.
    verbose : bool
        If True, print the reason for any failed lookup instead of
        silently swallowing it.

    Optional pre-fetched inputs (all keyword-only; each one skips a web request):
    yf_ticker : a yf.Ticker to reuse instead of creating a new one
    info : yfinance .info dict, used for the next earnings date
    headlines : [(date, title), ...] to scan for buyback/guidance keywords
        instead of calling Yahoo's news (e.g. Finnhub headlines)
    social_table : output of fetch_apewisdom_table()
    earnings_ratings : output of get_earnings_and_ratings() (e.g. from a cache)
    """
    today = dt.date.today()
    t = yf_ticker if yf_ticker is not None else yf.Ticker(ticker)
    report = CatalystReport(ticker=ticker.upper(), as_of=today)

    if earnings_ratings is None:
        earnings_ratings = get_earnings_and_ratings(ticker, yf_ticker=t, info=info,
                                                    ratings_lookback_days=ratings_lookback_days)

    # --- Earnings date proximity ---------------------------------------
    next_date = earnings_ratings.get("next_earnings_date")
    if next_date:
        report.next_earnings_date = next_date
        report.days_to_earnings = (next_date - today).days
        report.in_earnings_window = abs(report.days_to_earnings) <= 5

    # --- Index membership ------------------------------------------------
    members = (
        sp500_members if sp500_members is not None else _get_sp500_membership(verbose=verbose)
    )
    if members is not None:
        sym = ticker.upper()
        report.in_sp500 = sym in members
        if report.in_sp500:
            added = members.get(sym)
            if added:
                report.sp500_added_date = added
                report.days_since_index_addition = (today - added).days
                report.recent_index_addition = report.days_since_index_addition <= 60
    elif verbose:
        print("  [sp500 lookup] both sources failed")

    # --- News-derived: buybacks / guidance --------------------------------
    buyback_kw = ("buyback", "share repurchase", "repurchase program")
    guidance_kw = ("guidance", "outlook cut", "outlook raised", "forecast")
    cutoff = today - dt.timedelta(days=news_lookback_days)
    if headlines is None:  # no pre-fetched headlines -> Yahoo news
        headlines = []
        try:
            for item in t.get_news(count=25) or []:
                content = item.get("content", item)  # yfinance schema has shifted over versions
                title = (content.get("title") or "").strip()
                pub = content.get("pubDate") or content.get("providerPublishTime")
                pub_date = None
                if isinstance(pub, (int, float)):
                    pub_date = dt.datetime.fromtimestamp(pub).date()
                elif isinstance(pub, str):
                    try:
                        pub_date = dt.datetime.fromisoformat(pub.replace("Z", "+00:00")).date()
                    except ValueError:
                        pub_date = None
                headlines.append((pub_date, title))
        except Exception:
            pass

    for pub_date, title in headlines:
        if pub_date and pub_date < cutoff:
            continue
        lower = (title or "").lower()
        if any(k in lower for k in buyback_kw):
            report.buyback_headlines.append(title)
        if any(k in lower for k in guidance_kw):
            report.guidance_headlines.append(title)

    # --- Analyst upgrade/downgrade clustering ------------------------------
    for grade_date, firm, action_raw, grade in earnings_ratings.get("rating_rows", []):
        action = action_raw.lower()
        if "up" in action:
            report.upgrades += 1
        elif "down" in action:
            report.downgrades += 1
        report.rating_actions.append(f"{grade_date} {firm}: {action_raw} -> {grade}")

    # --- Social Sentiment (ApeWisdom) ----------------------------------
    social = get_apewisdom_sentiment(ticker, table=social_table)
    report.social_mentions = social.get("mentions", 0)
    report.social_upvotes = social.get("upvotes", 0)
    report.social_rank = social.get("rank", "N/A")
    report.social_mentions_24h_ago = social.get("mentions_24h_ago", 0)
    report.social_rank_24h_ago = social.get("rank_24h_ago", "N/A")
    report.social_momentum_pct = social.get("momentum_pct", "N/A")
    
    return report


def catalyst_snapshot(
    ticker: str,
    news_lookback_days: int = 14,
    ratings_lookback_days: int = 30,
    sp500_members: dict[str, dt.date] | None = None,
    as_dict: bool = True,
    verbose: bool = False,
) -> dict | CatalystReport:
    """
    One-call entry point: fetches everything and returns a plain dict
    (JSON-serializable, easy to drop into a DataFrame row or a scorecard)
    instead of the CatalystReport dataclass.

    Pass as_dict=False if you'd rather get the CatalystReport object back
    (e.g. to call .summary() for a printable version).

    Example
    -------
    >>> catalyst_snapshot("NVDA")
    {'ticker': 'NVDA', 'as_of': '2026-09-07', 'next_earnings_date': ...,
     'in_sp500': True, 'recent_index_addition': False, ...}
    """
    report = get_event_catalysts(
        ticker,
        news_lookback_days=news_lookback_days,
        ratings_lookback_days=ratings_lookback_days,
        sp500_members=sp500_members,
        verbose=verbose,
    )
    return report.to_dict() if as_dict else report


def catalyst_snapshots(
    tickers: list[str],
    news_lookback_days: int = 14,
    ratings_lookback_days: int = 30,
    verbose: bool = False,
) -> list[dict]:
    """
    Same as catalyst_snapshot but for a whole watchlist in one call.
    Fetches the S&P 500 membership list and the ApeWisdom ranking once and
    reuses them across all tickers instead of re-fetching per symbol.
    """
    members = _get_sp500_membership(verbose=verbose)
    social_table = fetch_apewisdom_table()
    results = []
    for tk in tickers:
        report = get_event_catalysts(
            tk,
            news_lookback_days=news_lookback_days,
            ratings_lookback_days=ratings_lookback_days,
            sp500_members=members,
            verbose=verbose,
            social_table=social_table,
        )
        results.append(report.to_dict())
    return results


if __name__ == "__main__":
    # Single ticker, one call:
    snap = catalyst_snapshot("NVDA", verbose=True)
    for k, v in snap.items():
        print(f"{k}: {v}")

    # Or, for the human-readable version:
    # print(get_event_catalysts("NVDA").summary())

    # Or for a whole watchlist in one call:
    # snaps = catalyst_snapshots(["NVDA", "MU", "ORCL"])
