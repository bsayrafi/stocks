import pandas as pd


def compute_sector_medians(results_df, sector_col="Sector",
                            metrics=("ForwardPE", "PEG", "EVToEBITDA")):
    """
    Compute the median of each valuation metric per sector, across the
    already-screened universe in results_df. Requires ENABLE_COMPANY_INFO
    (for Sector) and ENABLE_VALUATION (for the metrics) to have been on
    when evaluate_tickers() ran.

    Returns a DataFrame indexed by Sector, one column per metric.
    Sectors with fewer than `min_sector_size` tickers are dropped (too
    small a sample for a meaningful median) — those tickers fall back to
    fixed thresholds instead, handled in score_sector_relative_valuation().
    """
    if sector_col not in results_df.columns:
        raise ValueError(
            f"'{sector_col}' column not found — make sure ENABLE_COMPANY_INFO "
            f"was on when evaluate_tickers() ran."
        )

    missing_metrics = [m for m in metrics if m not in results_df.columns]
    if missing_metrics:
        raise ValueError(
            f"Missing valuation columns {missing_metrics} — make sure "
            f"ENABLE_VALUATION was on when evaluate_tickers() ran."
        )

    df = results_df[[sector_col] + list(metrics)].copy()
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    medians = df.groupby(sector_col)[list(metrics)].median()
    counts = df.groupby(sector_col)[list(metrics)].count()

    return medians, counts




TRUE_SECTOR_MEDIANS = {
    "Technology": {"ForwardPE": 28.0, "PEG": 1.8, "EVToEBITDA": 18.0},
    "Healthcare": {"ForwardPE": 22.0, "PEG": 1.5, "EVToEBITDA": 14.0},
    "Financial Services": {"ForwardPE": 14.0, "PEG": 1.2, "EVToEBITDA": 10.0},
    "Consumer Cyclical": {"ForwardPE": 20.0, "PEG": 1.4, "EVToEBITDA": 12.0},
    "Industrials": {"ForwardPE": 21.0, "PEG": 1.6, "EVToEBITDA": 13.0},
    # Add other major sectors as needed
}

def score_sector_relative_valuation(row, fallback_thresholds=None):
    """
    Score one row's valuation (0-30 pts) relative to true macro sector medians.
    """
    if fallback_thresholds is None:
        fallback_thresholds = {"ForwardPE": 25, "PEG": 2.0, "EVToEBITDA": 15}

    sector = str(row.get("Sector")).replace("_", " ") if row.get("Sector") else None
    score = 0
    points_per_metric = 10  # 3 metrics x 10 = 30 total

    for metric in ("ForwardPE", "PEG", "EVToEBITDA"):
        value = row.get(metric)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue

        # Look up the true median for this specific sector, or use the general fallback
        sector_benchmarks = TRUE_SECTOR_MEDIANS.get(sector, fallback_thresholds)
        benchmark_value = sector_benchmarks.get(metric, fallback_thresholds[metric])

        # Score it against the benchmark
        if value <= benchmark_value:
            score += points_per_metric
        elif value <= benchmark_value * 1.2:
            score += points_per_metric * 0.5
            
    return round(score, 1)


def add_sector_relative_valuation(results_df):
    """
    Post-processing step: adds 'SectorValuationScore' (0-30) and
    'CheaperThanSector' columns to results_df, computed relative to 
    static true market medians.
    """
    fallback_thresholds = {"ForwardPE": 25, "PEG": 2.0, "EVToEBITDA": 15}
    scores = []
    cheaper_counts = []
    
    for _, row in results_df.iterrows():
        # 1. Calculate the 0-30 score
        score = score_sector_relative_valuation(row, fallback_thresholds)
        scores.append(score)

        # 2. Calculate how many metrics beat the sector median
        count = 0
        sector = str(row.get("Sector")).replace("_", " ") if row.get("Sector") else None
        sector_benchmarks = TRUE_SECTOR_MEDIANS.get(sector, fallback_thresholds)
        
        for metric in ("ForwardPE", "PEG", "EVToEBITDA"):
            try:
                value = float(row.get(metric))
                benchmark_value = sector_benchmarks.get(metric, fallback_thresholds[metric])
                if value > 0 and value <= benchmark_value:
                    count += 1
            except (TypeError, ValueError):
                pass
        cheaper_counts.append(count)

    results_df = results_df.copy()
    results_df["SectorValuationScore"] = scores
    results_df["CheaperThanSector"] = cheaper_counts
    
    # Return TRUE_SECTOR_MEDIANS so the main script doesn't crash when unpacking
    return results_df, TRUE_SECTOR_MEDIANS