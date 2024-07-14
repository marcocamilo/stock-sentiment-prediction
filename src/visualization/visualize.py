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

#  ────────────────────────────────────────────────────────────────────
#   DUMP FROM KAGGLE                                                   
#  ────────────────────────────────────────────────────────────────────
stocks = pd.read_parquet('/kaggle/input/stock-history/stock-history-features.parquet')

index = 100
test = df.set_index('date')[:index]
closing_prices = stocks['close'][:index]
price_changes = closing_prices.diff().fillna(0)


scaler = MinMaxScaler(feature_range=(-1,1))
scaled_changes = scaler.fit_transform(price_changes.values.reshape(-1,1))
price_changes = pd.DataFrame(scaled_changes, columns=['close'])

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.4
positions = range(len(test))
ax.bar([pos - bar_width/2 for pos in positions], test['title_pos'], width=bar_width, label='Positive Sentiments', color='b', edgecolor='black')
ax.bar([pos + bar_width/2 for pos in positions], -test['title_neg'], width=bar_width, label='Negative Sentiments', color='r', edgecolor='black')

ax.plot(price_changes.index, price_changes['close'], 'g-o', label='Closing Prices')

ax.set_xlabel('Date')
ax.set_ylabel('Sentiment Score')
ax.set_title('Positive and Negative Sentiments Per Day')
ax.legend()
ax.set_xticks(positions)
ax.set_xticklabels('')
plt.tight_layout()

# Show plot
plt.show()
