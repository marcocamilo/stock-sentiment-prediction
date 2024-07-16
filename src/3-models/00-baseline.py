import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from src.modules.utils import evaluate_model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_parquet("./data/3-processed/stock-history-features.parquet")
df_full = pd.read_parquet("./data/3-processed/stock-history-features-full.parquet")

# ──────────────────────────────────────────────────────────────────── 
# BASELINE DATA: XGB, ROLLING MEAN 7, ROLLING MEAN 14
# ──────────────────────────────────────────────────────────────────── 
data = df
stocks = df.stock.unique()
results = dict()

for stock in stocks:
    main = data.query("stock == @stock")
    X, y = main.iloc[:, 2:-2].values, main.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    model = XGBRegressor()
    model.fit(X_train, y_train)
    xgb_pred = model.predict(X_test)
    
    rolling_mean_7 = main.iloc[-len(y_test):, -2].values
    rolling_mean_14 = main.iloc[-len(y_test):, -2].values
    true = y_test
    dates = main.iloc[-len(true):, 0].reset_index(drop=True)
    
    xgb_accuracy = (1 - mean_absolute_percentage_error(true, xgb_pred)) * 100
    rol7_accuracy = (1 - mean_absolute_percentage_error(true, rolling_mean_7)) * 100
    rol14_accuracy = (1 - mean_absolute_percentage_error(true, rolling_mean_14)) * 100
    
    results[stock] = {
        'dates': dates,
        'true': true,
        'xgb_pred': xgb_pred,
        'rolling_mean_7': rolling_mean_7,
        'rolling_mean_14': rolling_mean_14,
        'xgb_accuracy': round(xgb_accuracy, 2),
        'rol7_accuracy': round(rol7_accuracy, 2),
        'rol14_accuracy': round(rol14_accuracy, 2),
    }

# ──────────────────────────────────────────────────────────────────── 
# PLOT BASELINES 
# ──────────────────────────────────────────────────────────────────── 
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = axes.flatten()
for i, stock in enumerate(stocks):
    data = results[stock]
    
    # Plot Rolling Mean 7
    axes[i*3].plot(data['dates'], data['true'], label='True')
    axes[i*3].plot(data['dates'], data['rolling_mean_7'], label='Predicted')
    axes[i*3].set_title(f"{stock} – Rolling Mean 7: {data['rol7_accuracy']:.2f}%")
    axes[i*3].legend()
    
    # Plot Rolling Mean 14
    axes[i*3+1].plot(data['dates'], data['true'], label='True')
    axes[i*3+1].plot(data['dates'], data['rolling_mean_14'], label='Predicted')
    axes[i*3+1].set_title(f"{stock} – Rolling Mean 14: {data['rol14_accuracy']:.2f}%")
    axes[i*3+1].legend()
    
    # Plot XGB
    axes[i*3+2].plot(data['dates'], data['true'], label='True')
    axes[i*3+2].plot(data['dates'], data['xgb_pred'], label='Predicted')
    axes[i*3+2].set_title(f"{stock} – XGB: {data['xgb_accuracy']:.2f}%")
    axes[i*3+2].legend()

plt.tight_layout()
plt.show()

#  ────────────────────────────────────────────────────────────────────
#   EVALUATE MODEL
#  ────────────────────────────────────────────────────────────────────
for stock in df["stock"].unique():
    stock_df = df[df["stock"] == stock]
    print(stock.upper())
    evaluate_model(stock_df["close"], stock_df["rolling_mean_7"])
