import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

history = pd.read_parquet('./data/2-interim/stocks-history-cleaned.parquet')
news = pd.read_parquet('./data/2-interim/stock-news-cleaned.parquet')

#  ────────────────────────────────────────────────────────────────────
#   EXPLORE STOCKS                                                     
#  ────────────────────────────────────────────────────────────────────
sns.lineplot(history, x='Date', y='Close', hue='Stock')
plt.title(f"{history['Stock'][0]}: {history.Date.min()} – {history.Date.max()}")
plt.show()
