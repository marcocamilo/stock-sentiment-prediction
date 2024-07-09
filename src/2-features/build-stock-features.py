import pandas as pd
import pandas_ta as ta

df = pd.read_parquet("./data/2-interim/stock-history-cleaned.parquet")

#  ────────────────────────────────────────────────────────────────────
#   TIME FEATURES
#  ────────────────────────────────────────────────────────────────────
df["day_of_week"] = df["date"].dt.day_of_week
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

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
#   TECHNICAL INDICATORS
#  ────────────────────────────────────────────────────────────────────
df["rsi"] = df.groupby("stock")["close"].transform(lambda x: ta.rsi(x, length=14))
df["oversold"] = (df["rsi"] < 30).astype(int)
df["overbought"] = (df["rsi"] > 70).astype(int)

#  ────────────────────────────────────────────────────────────────────
#   SAVE DATA
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("./data/3-processed/stock-history-features.parquet", index=False)
