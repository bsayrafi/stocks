# stocks

Screeners, strategy research and live signals.

## Run things from this folder

The scripts you start live right here in the project root:

| Script | What it runs | Also started by |
|---|---|---|
| `run_core_screener.py` | v1 daily screener (Excel + HTML report) | workflow `run_core_screener` |
| `run_intraday_vwap.py` | v1 screener + intraday VWAP levels | workflow `run_intraday` |
| `run_buy_entry.py` | v1 screener + buy-zone / dip / trend confirmation | workflow `run_buy_signal` |
| `run_trend_entry.py` | v2 setup screener (`enrich_html_v2`) | workflow `run_html_v2`, the local report server |
| `run_backtest_html2.py` | walk-forward backtest of the v2 setups | by hand |
| `run_backtest_flow.py` | backtest of the dip/trend accumulation scores | by hand |
| `run_mds_scanner.py` | MDS live scanner (`entry` / `daily`) | `run_mds.sh` (launchd) |
| `run_scan.py` | sector rotation + macro regime scan | workflow `check_sectors` |
| `run_local_*.sh` | any script with the API keys from `.env` | by hand |
| `run_mds.sh`, `run_tsm_live_signal.sh` | launchd wrappers | launchd on the Mac |

    python run_trend_entry.py refresh
    ./run_local_buyentry.sh                        # run_buy_entry.py with keys from .env
    ./run_local_buyentry.sh fvg_live_signal.py     # any other script, found in its folder by name

Scripts inside the folders run from here too, either way:

    python -m fvg.fvg_live_signal --sector Technology
    python fvg/fvg_live_signal.py --sector Technology

Data folders stay at the root and are created on demand: `data/`, `cache/`, `models/`, `reports/`, `results/`,
`live/`, `logs/` (most are git-ignored).

## The code

| Folder | What it is |
|---|---|
| `common/` | Shared helpers: config (`constants`), indicators, caches, ticker lists, benchmarks, earnings/event catalysts, macro calendar, intraday VWAP levels, survivorship-free universe |
| `screener/` | v1 daily multi-timeframe screener (`enrich_html`, `screener`, `buy_entry`) |
| `setups_v2/` | v2 setup engine (`enrich_html_v2`), walk-forward backtest, portfolio simulator, local report server (`setups_v2/install_report_server.sh` on the Mac) |
| `confirm/` | Accumulation confirmation on Alpaca 30-min bars (`dip_confirm`, `trend_confirm`) and its backtest |
| `fvg/` | Hourly Fair Value Gap ML pipeline: dataset, model, live signal (workflow `run_fvg_signal`), fill analyses. `rule_backtest/` = older rule-based FVG backtester |
| `trend_tsm/` | Hourly trend model and live signal (launchd `com.bassem.tsm-live-signal`, 07:00) |
| `mds/` | MDS strategy, config, data, backtests, report (see `mds/MDS_STRATEGY.md`; launchd plists here) |
| `macro/` | Sector rotation and macro regime |
| `tools/` | One-off checks (Alpaca SIP access) |
| `archive/` | Old scripts nothing uses (Colab-era `analyze`/`fetch_data`, demos). Safe to delete |
