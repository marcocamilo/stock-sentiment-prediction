import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

history = pd.read_parquet('./data/2-interim/stocks-history-cleaned.parquet')
news = pd.read_parquet('./data/2-interim/stock-news-cleaned.parquet')

#  ────────────────────────────────────────────────────────────────────
#   EXPLORE STOCKS                                                     
#  ────────────────────────────────────────────────────────────────────
sns.lineplot(history, x='date', y='close', hue='stock')
plt.title(f"{history['stock'][0]}: {history.date.min()} – {history.date.max()}")
plt.show()
