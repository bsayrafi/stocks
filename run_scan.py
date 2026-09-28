"""
run_scan.py  -  run the full scan and save results to ./data/

    python run_scan.py

Keep this file in the same folder as sector_rotation.py, macro_regime.py, market_scan.py.
"""

import pandas as pd

from market_scan import scan_market

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)     # show every column, no "..."
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda v: f"{v:.2f}")

res = scan_market(
    universe="sectors+tech",
    benchmark="SPY",
    baskets={
        "AI compute": ["NVDA", "AVGO", "AMD", "TSM"],
        "AI apps":    ["MSFT", "PLTR", "CRM", "NOW"],
    },
)

print()
print(res["table"])
print(f"\nCSV: {res['csv_path']}")
