import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

df = pd.read_parquet("./data/2-interim/stock-history-cleaned.parquet")

#  ────────────────────────────────────────────────────────────────────
#   DOWNLOAD LOOKBACK DATA                                                       
#  ────────────────────────────────────────────────────────────────────
stocks = ["TSLA", "AAPL", "GOOG"]
for stock in stocks:
    data = yf.download(stock, start="2021-12-22", end="2022-01-03", ignore_tz=False).reset_index()
    data.columns = [col.lower() for col in data.columns]
    data = data[['date', 'close']]
    data["stock"] = stock
    df = pd.concat([df, data], axis=0)
df = df.sort_values(["stock", "date"]).reset_index(drop=True)

#  ────────────────────────────────────────────────────────────────────
#   TIME FEATURES
#  ────────────────────────────────────────────────────────────────────
df["day_of_week"] = df["date"].dt.day_of_week
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

#  ────────────────────────────────────────────────────────────────────
#   LAG FEATURES                                                       
#  ────────────────────────────────────────────────────────────────────
df["lag_1"] = df.groupby("stock")["close"].shift(1)
df["lag_2"] = df.groupby("stock")["close"].shift(2)
df["lag_3"] = df.groupby("stock")["close"].shift(3)
df["lag_4"] = df.groupby("stock")["close"].shift(4)
df["lag_5"] = df.groupby("stock")["close"].shift(5)
df["lag_6"] = df.groupby("stock")["close"].shift(6)
df["lag_7"] = df.groupby("stock")["close"].shift(7)

#  ────────────────────────────────────────────────────────────────────
#   ROLLING STATISTICS
#  ────────────────────────────────────────────────────────────────────
df["rolling_mean_7"] = df.groupby("stock")["close"].transform(
    lambda x: x.rolling(window=7).mean()
)
df["rolling_std_7"] = df.groupby("stock")["close"].transform(
    lambda x: x.rolling(window=7).std()
)
df["rolling_mean_14"] = df.groupby("stock")["close"].transform(
    lambda x: x.rolling(window=14, min_periods=7).mean()
)
df["rolling_std_14"] = df.groupby("stock")["close"].transform(
    lambda x: x.rolling(window=14, min_periods=7).std()
)

#  ────────────────────────────────────────────────────────────────────
#   CLEAN DATA                                                         
#  ────────────────────────────────────────────────────────────────────
df = df.dropna().reset_index(drop=True)
df.drop("date", axis=1, inplace=True)
df.insert(16, "close", df.pop("close"))

#  ────────────────────────────────────────────────────────────────────
#   FEATURE IMPORTANCE                                                 
#  ────────────────────────────────────────────────────────────────────
stocks = dict(
    tsla = df.query("stock == 'TSLA'").copy().values,
    aapl = df.query("stock == 'AAPL'").copy().values,
    goog = df.query("stock == 'GOOG'").copy().values,
)

feature_importance = dict()

for key, data in stocks.items():
    model = RandomForestRegressor()
    X = data[:,1:-1]
    y = data[:,-1]
    model.fit(X, y)
    feature_importance[key] = model.feature_importances_

columns = df.columns.to_list()[1:-1]

fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ax, (key, data) in zip(axes, feature_importance.items()):
    sns.barplot(x=columns, y=data, ax=ax)
    ax.set(ylim=(0, 1))
    ax.set_title(key)
    ax.tick_params(labelrotation=45)
plt.tight_layout()
plt.show()

#  ────────────────────────────────────────────────────────────────────
#   DROP LESS IMPORTANT FEATURES                                       
#  ────────────────────────────────────────────────────────────────────
df.drop(columns=[
    "lag_3", "lag_4", "lag_5", "lag_6", "lag_7",
    "rolling_std_7", "rolling_std_14",
], inplace=True)

#  ────────────────────────────────────────────────────────────────────
#   SAVE DATA
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("./data/3-processed/stock-history-features.parquet", index=False)
