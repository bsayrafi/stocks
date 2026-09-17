"""
inspect_hour_of_day.py
========================
Diagnostic: FVG success rate broken down by formation hour, to see whether
"hour_of_day" is capturing a real session-timing effect (e.g. open-hour
volatility vs. midday chop vs. power-hour follow-through) or just noise
that happened to get some tree splits.

Usage:
    python3 inspect_hour_of_day.py --data data/fvg_hourly_tech.parquet
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fvg_hourly_tech.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=["label"])

    by_hour = df.groupby("hour_of_day").agg(
        n=("label", "size"),
        success_rate=("label", "mean"),
    ).round(3)
    print("By hour of day (market-local, America/New_York):")
    print(by_hour)

    print(f"\nOverall base rate: {df['label'].mean():.3f}")

    # split by direction too, in case bullish/bearish FVGs behave differently by session
    print("\nBy hour of day x direction:")
    by_hour_dir = df.groupby(["hour_of_day", "direction"]).agg(
        n=("label", "size"),
        success_rate=("label", "mean"),
    ).round(3)
    print(by_hour_dir)


if __name__ == "__main__":
    main()
