import pandas as pd
df = pd.read_parquet("data/fvg_hourly_tech.parquet")
print(df["htf_trend_dir"].value_counts())
print(df["htf_confluence"].value_counts())