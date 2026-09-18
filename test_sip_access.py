"""
test_sip_access.py
=====================
Tests a hypothesis: Alpaca's free tier restricts SIP data only for the
LAST 15 MINUTES (per their own plan comparison table), not historical
depth in general. Our existing pipeline always requests up to
datetime.utcnow(), which touches that restricted window and forces us
onto feed='iex' for the entire range. This script requests the SAME
historical range twice — once via SIP, once via IEX — but with `end`
set safely outside the last 15 minutes, to see whether SIP succeeds on
the free tier when the recency restriction isn't triggered, and whether
the two feeds' VOLUME actually differs (proof SIP is capturing more of
the tape, not just a successful call).

Usage:
    python3 test_sip_access.py --symbol AAPL
"""

import os
import argparse
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()


def try_fetch(client, symbol, feed, end, start):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
        adjustment="all",
        feed=feed,
    )
    try:
        bars = client.get_stock_bars(req).df
        if bars is None or bars.empty:
            return None, "empty result (no error, but no rows — check date range / market hours)"
        return bars, None
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--minutes-buffer", type=int, default=30,
                         help="How far before 'now' to set the end time, to safely clear the 15-min restricted window.")
    args = parser.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise SystemExit("Set ALPACA_API_KEY / ALPACA_SECRET_KEY in your .env file.")

    client = StockHistoricalDataClient(api_key, api_secret)

    end = datetime.now(timezone.utc) - timedelta(minutes=args.minutes_buffer)
    start = end - timedelta(days=5)

    print(f"Requesting {args.symbol} hourly bars: {start} -> {end}  "
          f"(end is {args.minutes_buffer} min before 'now', to clear the 15-min restricted window)\n")

    print("--- Trying feed=SIP ---")
    sip_bars, sip_err = try_fetch(client, args.symbol, DataFeed.SIP, end, start)
    if sip_err:
        print(f"FAILED: {sip_err}")
    else:
        print(f"SUCCESS: {len(sip_bars)} bars retrieved.")
        print(sip_bars[["open", "high", "low", "close", "volume"]].tail(5))

    print("\n--- Trying feed=IEX (for comparison) ---")
    iex_bars, iex_err = try_fetch(client, args.symbol, DataFeed.IEX, end, start)
    if iex_err:
        print(f"FAILED: {iex_err}")
    else:
        print(f"SUCCESS: {len(iex_bars)} bars retrieved.")
        print(iex_bars[["open", "high", "low", "close", "volume"]].tail(5))

    if sip_bars is not None and iex_bars is not None:
        print("\n--- Volume comparison (same timestamps, SIP vs IEX) ---")
        common_idx = sip_bars.index.intersection(iex_bars.index)
        if len(common_idx) == 0:
            print("No overlapping timestamps to compare (unexpected — check index format).")
        else:
            sip_vol = sip_bars.loc[common_idx, "volume"]
            iex_vol = iex_bars.loc[common_idx, "volume"]
            ratio = (iex_vol.sum() / sip_vol.sum()) if sip_vol.sum() > 0 else float("nan")
            print(f"Total SIP volume: {sip_vol.sum():,.0f}")
            print(f"Total IEX volume: {iex_vol.sum():,.0f}")
            print(f"IEX as fraction of SIP: {ratio:.1%}")
            print("\n(If IEX is meaningfully less than SIP — commonly IEX carries roughly "
                  "1-3% of consolidated volume — that confirms SIP is genuinely capturing "
                  "more of the tape, not just returning success with the same data.)")


if __name__ == "__main__":
    main()
