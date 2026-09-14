def calculate_dip_pct(df, days, lookback_days=None):
    """
    % decline of D1 close from the highest close in the last `lookback_days`
    (including D1 itself). Positive value = dip magnitude; 0 = at/above recent high.
    """
    from constants import CONFIG, DAYm1_IDX, DAY_IDXS
    lookback_days = lookback_days if lookback_days is not None else CONFIG["DIP_LOOKBACK_DAYS"]

    recent_closes = df['Close'].iloc[-lookback_days:]
    recent_high = recent_closes.max()
    d1_close = days[DAYm1_IDX]['Close']

    if recent_high == 0:
        return 0.0
    return round(max(0.0, (recent_high - d1_close) / recent_high) * 100, 2)


def is_dip_candidate(dip_pct, min_pct=None):
    from constants import CONFIG
    min_pct = min_pct if min_pct is not None else CONFIG["DIP_MIN_PCT"] * 100
    return 1 if dip_pct >= min_pct else 0


def is_near_earnings(earnings_dates_df, exclusion_days=None):
    """
    Returns 1 if today falls within `exclusion_days` of any earnings date
    (past or upcoming) in the earnings_dates DataFrame from yfinance
    (get_earnings_dates()). Returns 0 if clear, or if no earnings data available
    (fails safe toward NOT excluding, since missing data shouldn't silently
    disqualify a candidate — flag this in output instead, see below).
    """
    from constants import CONFIG
    import pandas as pd

    exclusion_days = exclusion_days if exclusion_days is not None else CONFIG["EARNINGS_EXCLUSION_DAYS"]

    if earnings_dates_df is None or earnings_dates_df.empty:
        return 0  # unknown — treat as not excluded, but caller should note data was missing

    today = pd.Timestamp.now(tz=earnings_dates_df.index.tz) if earnings_dates_df.index.tz else pd.Timestamp.now()

    for edate in earnings_dates_df.index:
        days_away = abs((edate - today).days)
        if days_away <= exclusion_days:
            return 1
    return 0


def calculate_base_quality_score(cash_metrics, altman_z, analyst_data, eps_data):
    """
    Cash + solvency + analyst + EPS/growth (0-70 pts). Deliberately EXCLUDES
    valuation — see combine_final_quality_scores() for how the full 100-pt
    score (base + sector-relative valuation) is assembled.
    """
    score = 0

    # --- Cash health (up to 18 pts) ---
    if cash_metrics:
        if cash_metrics.get("FCFYield") is not None and cash_metrics["FCFYield"] > 0:
            score += 6
        if cash_metrics.get("CurrentRatio") is not None and cash_metrics["CurrentRatio"] >= 1.2:
            score += 6
        if cash_metrics.get("CashToDebt") is not None and cash_metrics["CashToDebt"] >= 0.5:
            score += 6

    # --- Solvency (up to 13 pts) ---
    if altman_z is not None:
        if altman_z >= 2.99:
            score += 13
        elif altman_z >= 1.81:
            score += 7

    # --- Analyst backing (up to 17 pts) ---
    if analyst_data:
        rec_mean = analyst_data.get("RecMean")
        if rec_mean is not None and rec_mean <= 2.5:
            score += 9
        buy_count = analyst_data.get("BuyCount")
        sell_count = analyst_data.get("SellCount")
        if buy_count is not None and sell_count is not None and buy_count > sell_count:
            score += 8

    # --- EPS revision direction + magnitude (up to 12 pts) ---
    if eps_data:
        up7 = eps_data.get("EPSRevUp7d") or 0
        down7 = eps_data.get("EPSRevDown7d") or 0
        if up7 >= down7:
            score += 6

        current = eps_data.get("EPSTrendCurrent")
        ago_30d = eps_data.get("EPSTrend30dAgo")

        if current is not None and ago_30d not in (None, 0):
            try:
                pct_change = (float(current) - float(ago_30d)) / abs(float(ago_30d)) * 100
                if pct_change > 0:
                    score += 6
                elif pct_change >= -2:
                    score += 3
                elif pct_change <= -5:
                    # Consensus EPS estimate has been slashed by 5%+ in the last month.
                    # Apply toxic penalty to override historical quarterly data.
                    score -= 15
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    # --- Growth (up to 10 pts) — durability-weighted: reward growth that
    # holds up across both near-term (quarter) and longer-term (full year)
    # horizons, not just a single quarter's estimate ---
    if eps_data:
        q_growth = eps_data.get("EPSEstGrowth")
        fy_growth = eps_data.get("EPSEstGrowthFY")

        try:
            q_growth = float(q_growth) if q_growth is not None else None
        except (TypeError, ValueError):
            q_growth = None
        try:
            fy_growth = float(fy_growth) if fy_growth is not None else None
        except (TypeError, ValueError):
            fy_growth = None

        if q_growth is not None and fy_growth is not None:
            # Durable growth: both quarter and full-year estimates are positive
            if q_growth > 0 and fy_growth > 0:
                score += 10
            elif q_growth > 0 or fy_growth > 0:
                score += 4  # growth in one horizon only — less durable, partial credit
        elif q_growth is not None and q_growth > 0:
            score += 5  # only quarterly data available
        elif fy_growth is not None and fy_growth > 0:
            score += 5  # only full-year data available

    return score  # max 70



def is_quality_dip_buy(dip_pct, near_earnings, quality_score, min_dip_pct=None, min_quality=None):
    """
    Final flag: a real dip, NOT earnings-driven, in a fundamentally sound company.
    """
    from constants import CONFIG
    min_dip_pct = min_dip_pct if min_dip_pct is not None else CONFIG["DIP_MIN_PCT"] * 100
    min_quality = min_quality if min_quality is not None else CONFIG["DIP_QUALITY_MIN_SCORE"]

    return 1 if (dip_pct >= min_dip_pct and near_earnings == 0 and quality_score >= min_quality) else 0


def combine_final_quality_scores(results_df):
    """
    Call AFTER add_sector_relative_valuation() has added 'SectorValuationScore'.
    Combines BaseQualityScore (cash/solvency/analyst/EPS, 0-70) with
    SectorValuationScore (0-30) into the final QualityScore (0-100), then
    recomputes QualityDipBuy using that final score. This is the ONLY place
    the two components should be added together — avoids double-counting.
    """
    from constants import CONFIG
    import pandas as pd

    results_df = results_df.copy()

    base = pd.to_numeric(results_df.get("BaseQualityScore"), errors="coerce").fillna(0)
    sector_val = pd.to_numeric(results_df.get("SectorValuationScore"), errors="coerce").fillna(0)

    results_df["QualityScore"] = (base + sector_val).round(1)

    min_dip_pct = CONFIG["DIP_MIN_PCT"] * 100
    min_quality = CONFIG["DIP_QUALITY_MIN_SCORE"]

    def recompute_flag(row):
        dip_pct = row.get("DipPct")
        near_earnings = row.get("NearEarnings")
        quality_score = row.get("QualityScore")
        if dip_pct is None or near_earnings is None or quality_score is None:
            return 0
        return 1 if (dip_pct >= min_dip_pct and near_earnings == 0 and quality_score >= min_quality) else 0

    results_df["QualityDipBuy"] = results_df.apply(recompute_flag, axis=1)
    return results_df
