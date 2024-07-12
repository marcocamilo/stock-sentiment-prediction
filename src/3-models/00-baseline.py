import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from src.modules.utils import evaluate_model

df = pd.read_parquet("./data/3-processed/stock-history-features.parquet")

#  ────────────────────────────────────────────────────────────────────
#   MOVING AVERAGE
#  ────────────────────────────────────────────────────────────────────
# fig = go.Figure()
# for stock in df["stock"].unique():
#     stock_df = df[df["stock"] == stock]
#     fig.add_trace(
#         go.Scatter(
#             x=np.arange(len(stock_df)), y=stock_df["close"], mode="lines", name=stock
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=np.arange(len(stock_df)), y=stock_df["rolling_mean_7"], mode="lines", name=f"{stock} - rolling_mean_7"
#         )
#     )
# fig.update_layout(title="Stocks Closing Price", xaxis_title="Date", yaxis_title="Price")
# fig.show()

fig, axes = plt.subplots(3,1, figsize=(20,5))
axes = axes.flatten()
for ax, stock in zip(axes, df["stock"].unique()):
    stock_df = df[df["stock"] == stock]
    ax.plot(stock_df["close"], label=stock)
    ax.plot(stock_df["rolling_mean_7"], label=f"{stock} - rolling_mean_7")
    ax.set_title(f"{stock}: Closing Price vs Moving Average (7 days)")
plt.tight_layout()
plt.show()

#  ────────────────────────────────────────────────────────────────────
#   EVALUATE MODEL
#  ────────────────────────────────────────────────────────────────────
for stock in df["stock"].unique():
    stock_df = df[df["stock"] == stock]
    print(stock.upper())
    evaluate_model(stock_df["close"], stock_df["rolling_mean_7"])
