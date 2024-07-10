import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("./data/2-interim/stock-history-cleaned.parquet")

#  ────────────────────────────────────────────────────────────────────
#   TIME FEATURES
#  ────────────────────────────────────────────────────────────────────
df["day_of_week"] = df["date"].dt.day_of_week
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

df.drop("date", axis=1, inplace=True)

#  ────────────────────────────────────────────────────────────────────
#   ROLLING STATISTICS
#  ────────────────────────────────────────────────────────────────────
df["rolling_mean_7"] = df.groupby("stock")["close"].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
df["rolling_std_7"] = df.groupby("stock")["close"].transform(
    lambda x: x.rolling(window=7, min_periods=1).std()
)
df["rolling_mean_14"] = df.groupby("stock")["close"].transform(
    lambda x: x.rolling(window=14, min_periods=1).mean()
)
df["rolling_std_14"] = df.groupby("stock")["close"].transform(
    lambda x: x.rolling(window=14, min_periods=1).std()
)

#  ────────────────────────────────────────────────────────────────────
#   CLEAN DATA                                                         
#  ────────────────────────────────────────────────────────────────────
df = df.dropna()

#  ────────────────────────────────────────────────────────────────────
#   SAVE DATA
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("./data/3-processed/stock-history-features.parquet", index=False)
