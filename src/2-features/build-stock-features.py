import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor

# df = pd.read_parquet("./data/2-interim/stock-history-cleaned.parquet")
df = pd.read_parquet("~/Downloads/stock-history-features-full.parquet")

#  ────────────────────────────────────────────────────────────────────
#   TIME FEATURES
#  ────────────────────────────────────────────────────────────────────
df["day_of_week"] = pd.to_datetime(df["date"]).dt.day_of_week
df["day"] = pd.to_datetime(df["date"]).dt.day
df["month"] = pd.to_datetime(df["date"]).dt.month
df["year"] = pd.to_datetime(df["date"]).dt.year

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
start_date = pd.to_datetime("2022-03-01").date()
end_date = pd.to_datetime("2024-07-01").date()
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
df = df[mask].reset_index(drop=True)

df.insert(17, "close", df.pop("close"))

#  ────────────────────────────────────────────────────────────────────
#   FEATURE IMPORTANCE
#  ────────────────────────────────────────────────────────────────────
stocks = dict(
    tsla=df.query("stock == 'TSLA'").copy().values,
    aapl=df.query("stock == 'AAPL'").copy().values,
    goog=df.query("stock == 'GOOG'").copy().values,
)

feature_importance = dict()

for key, data in stocks.items():
    model = XGBRegressor()
    X = data[:, 2:-1]
    y = data[:, -1]
    model.fit(X, y)
    feature_importance[key] = model.feature_importances_

columns = df.columns.to_list()[2:-1]

fig, axes = plt.subplots(3, 1)
axes = axes.flatten()

for ax, (key, data) in zip(axes, feature_importance.items()):
    sns.barplot(x=columns, y=data, ax=ax)
    ax.set(ylim=(0, 1))
    ax.set_title(key)
    ax.tick_params(labelrotation=45)
    for i, v in enumerate(data):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()

#  ────────────────────────────────────────────────────────────────────
#   DROP LESS IMPORTANT FEATURES
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("./data/3-processed/stock-history-features-full.parquet")

# df.drop(
#     columns=[
#         "lag_3",
#         "lag_4",
#         "lag_5",
#         "lag_6",
#         "lag_7",
#         "rolling_std_7",
#         "rolling_std_14",
#     ],
#     inplace=True,
# )

df = df[['lag_1', 'rolling_mean_7']]

#  ────────────────────────────────────────────────────────────────────
#   SAVE DATA
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("./data/3-processed/stock-history-features-minimal.parquet")
