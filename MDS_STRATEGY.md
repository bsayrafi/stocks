# MDS — Multi-Day Swing strategy

Long-only, 3–15 trading-day holds. Nothing is predicted: every gate is a *state* (trend is up, money is flowing in, the sector is being rotated into), and the trade is managed by structure (support / resistance / volume nodes).

## Files

| File | Role |
|---|---|
| `mds_config.py` | `MDS_CONFIG` — every threshold in one dict |
| `mds_strategy.py` | the rules (shared by scanner and backtest) |
| `mds_data.py` | yfinance / Alpaca loaders, finviz + survivorship-free universes, backtest cache |
| `mds_report.py` | HTML report (charts, sector map, levels) |
| `run_mds_scanner.py` | `daily` scan → watchlist + report + ntfy; `entry` 15m trigger → ntfy alert |
| `mds_backtest.py` | `cache` / `simulate` backtest on Alpaca SIP daily + 15m bars |
| `run_mds.sh`, `com.bassem.mds-*.plist` | launchd automation on the Mac |
| `.github/workflows/run_mds_scanner.yml` | optional cloud daily report |

Reused from the existing code: `boxindicators.atr/adx`, `indicators.calculate_support_resistance`, `intraday_strategy.compute_volume_profile` + `SECTOR_TO_ETF`, `cached_ticker.CachedTicker`, `enrich_html.send_ntfy_file`, `survivorship_free_universe`. `rolling_lrc` reproduces `enrich_html.linear_regression_channel` exactly (self-test: `python3 mds_strategy.py`), and the vectorized session VWAP matches `intraday_strategy.add_vwap_bands` to 1e-13.

## Rules

| # | Pillar | Rule (default) |
|---|---|---|
| 0 | Market | SPY close > SMA200, otherwise no new longs |
| 1 | Sector rotation | stock's SPDR ETF in the **Improving** RRG quadrant (RS-Ratio = RS / SMA50(RS) < 100, RS-Momentum = 10-day change of RS-Ratio > 100) — money rotating *into* a sector that has lagged |
| 2 | Fundamentals | Yahoo rating `buy`/`strong_buy` with ≥5 analysts · beta ≤ 2 · 0 < PEG < 4 · ≥3/5 quality (margin, revenue growth, FCF, EPS revisions, D/E) · no earnings within 15 days |
| 3 | Trend | 50-day regression channel slope > 0 · price in the lower 75% of the channel (R² ≥ 0.5, EMA21 > EMA50 and ADX ≥ 20 are available as switches but off — see backtest) |
| 4 | Institutional flow | close > VWAP anchored at the channel's lowest low · ≥2/3 of: up/down volume (50d) > 1, CMF(20) > 0, OBV rising |
| 5 | Levels | composite 20-session 15m volume profile (POC/VAH/VAL) + daily fractal swings. **Stop** = highest support ≥0.75 ATR below − 0.25 ATR (max 3 ATR). **TP1** = nearest daily resistance ≥1 ATR above (swing high / 52w high / channel top; POC only when price is below value). **TP2** = next one ≥1 ATR beyond TP1 |
| 6 | Buy zone | `max_entry = (TP1 + 2·stop) / 3` — the highest price with R:R ≥ 2. Must lie within 1 ATR below the close |
| 7 | 15m entry (next 3 sessions) | green bar, close > EMA9, time-of-day RVOL ≥ 1, R:R ≥ 2 at the bar close, and either **VWAP reclaim** (touched/reclaimed session VWAP, ≤ VWAP+1σ) or **level bounce** (tagged a daily support within 0.2 ATR, close ≥ VWAP−1σ). Void if price hits the stop first; skip sessions that gap > 1 ATR up |
| 8 | Management | sell 50% at TP1, stop → breakeven; rest at TP2, on a daily close below the channel's lower band, or after 15 sessions |

## Running

```bash
# after the close (or let com.bassem.mds-daily.plist do it)
./run_mds.sh daily --size 2          # report in reports/MDS_*.html, watchlist in live/
# during the session every 15 min (com.bassem.mds-entry.plist)
./run_mds.sh entry

# backtest
python3 mds_backtest.py cache --universe sp500-pit --daily-years 5 --intraday-years 3
python3 mds_backtest.py simulate                     # technical + flow + sector rules
python3 mds_backtest.py simulate --no-sector         # ablation: is rotation adding anything?
python3 mds_backtest.py simulate --fundamentals-snapshot   # look-ahead! comparison only
```

## Daily-bar backtest (2019-10 → 2026-09)

`mds_backtest_daily.py` on `cache/tsm_daily` — Alpaca SIP daily bars, ~370 S&P 500 Tech / Health Care / Financials names including ones later removed from the index. SPY and sector ETFs replaced by equal-weight stand-ins; 15m trigger replaced by a limit at the buy zone; no fundamentals; 5 bps/side; 1% risk per trade. IS = 2019–2023, OOS = 2024–2026.

| Variant | Trades | Win | Avg R | IS avg R | OOS avg R | CAGR | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|---|
| Original spec (Leading+Improving, R² ≥ 0.5, EMA, ADX, flow) | 5,642 | 34% | +0.03 | +0.03 | +0.02 | −3.4% | −48% | −0.19 |
| Control: levels + exits only, no filters | 48,015 | 35% | +0.04 | +0.04 | +0.03 | +6.9% | −18% | 0.49 |
| **Current default** (Improving, slope > 0, flow + AVWAP) | 4,303 | 39% | **+0.19** | +0.19 | +0.18 | **+11.1%** | **−9%** | **1.13** |
| Same, sector gate off | 19,696 | 34% | +0.02 | +0.03 | +0.02 | +7.6% | −16% | 0.55 |
| Equal-weight buy & hold of the universe | — | — | — | — | — | +20.4% | −39% | 0.90 |

Findings: the sector filter carries the edge (Improving +0.18R, Leading −0.02R, Weakening −0.04R, in both periods); R², EMA stack and ADX added nothing; strong recent accumulation (CMF/OBV top quintile) was *worse* for 3–15-day pullback entries. The Improving edge holds across RRG windows 40–60 / momentum 5–12 (avg R +0.16…+0.23) and fades at 75–100. Losing year: 2022 (−7.5% vs −14.8% for buy & hold).

## Real-data check (15m, Alpaca SIP, real SPY + 11 sector ETFs)

`mds_backtest.py simulate` on the survivorship-free S&P 500 cache (first 329 names alphabetically, Oct 2023 → Sep 2026; SPY +25.5%/yr over the same window):

| Variant | Trades | Avg R | CAGR | Max DD | Sharpe |
|---|---|---|---|---|---|
| Current default (Improving, VWAP reclaim + level bounce, TP1/TP2) | 771 | +0.06 | −2.3% | −23% | −0.15 |
| VWAP reclaim only | 604 | +0.10 | +0.7% | −22% | 0.12 |
| VWAP reclaim, no fixed targets, 3-ATR trailing stop, 40-session max | 588 | +0.13 | +4.6% | −19% | 0.41 |
| Same, sector filter off | 3,046 | +0.17 | +4.1% | −19% | 0.33 |

The "Improving sector" edge seen with equal-weight stand-in baskets did not carry over to the real ETFs. The level-bounce entry lost money (−0.09R); fixed targets capped the winners. Higher R², higher rank score and stronger trends were *worse*, not better, at this 3–15-day horizon.
Switches for these variants: `--set ENTRY_TYPES=vwap_reclaim USE_TARGETS=0 TRAIL_ATR=3 MAX_HOLD_SESSIONS=40` (mds_backtest.py) and `XP_TP=0 XP_TRAIL_ATR=3` (mds_backtest_daily.py).

## Backtest caveats

- Analyst ratings, beta and PEG have no free point-in-time history → off by default; `--fundamentals-snapshot` applies *today's* gate to all history (look-ahead).
- Finviz universes are today's survivors; use `--universe sp500-pit`.
- Stop and target touched in the same 15m bar → stop assumed first. Gaps fill at the open. 5 bps per side.
- Phase A builds each symbol's trades independently; phase B applies capital, 8 positions max, 3 per sector, 1% equity risk per trade, 20% max per position.

## Quality Dip test (`qd_backtest.py`) — short-term reversal

Rules fixed before testing: SPY > SMA200, stock > SMA200, RSI(2) < 10 → buy next open, sell on the first close above SMA(5) or after 10 sessions. Variant B adds CMF(20) < 0. 20 slots × 5% equity.

| Data | Variant | Trades | Win | Avg/trade | SPY over the same days | **Excess vs SPY** | Portfolio CAGR | SPY CAGR |
|---|---|---|---|---|---|---|---|---|
| 8y Tech/HC/Fin (EW index) | A | 12,848 | 65% | +0.28% | +0.35% | −0.08% | +2.5% | +18.7% |
| 8y Tech/HC/Fin (EW index) | B | 7,431 | 66% | +0.39% | +0.40% | −0.01% | +4.0% | +18.7% |
| 5y all sectors, real SPY | A | 9,557 | 64% | +0.21% | +0.31% | −0.10% (t −2.8) | +5.9% | +17.9% |
| 5y all sectors, real SPY | B | 5,431 | 63% | +0.18% | +0.29% | −0.11% (t −2.3) | +6.0% | +17.9% |

The dip trades win often, but only because the whole market bounces after sell-offs: holding SPY over the same days did as well or better. No stock-specific edge. `excess_vs_index.py` (trade return minus index return over the same holding window) should be the pass/fail test for any future idea.
