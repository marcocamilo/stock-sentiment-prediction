import pandas as pd
from tqdm import tqdm

#  ────────────────────────────────────────────────────────────────────
#   CREATE MASTER DATASET                                              
#  ────────────────────────────────────────────────────────────────────
stocks = ["TSLA", "AAPL", "GOOG"]

df = pd.concat(
    [pd.read_parquet(f'./data/1-raw/{stock}/{stock.lower()}-history.parquet').reset_index().assign(stock=stock)
     for stock in tqdm(stocks)], ignore_index=True
)
print(df.shape)
print(df.dtypes)

#  ────────────────────────────────────────────────────────────────────
#   DUPLICATE AND MISSING VALUES CHECK                                 
#  ────────────────────────────────────────────────────────────────────
df.isna().mean()
df.duplicated().mean()

#  ────────────────────────────────────────────────────────────────────
#   KEEP RELEVANT COLUMNS                                              
#  ────────────────────────────────────────────────────────────────────
df = df[['Date', 'Close', 'stock']]
df.columns = df.columns.str.lower()

#  ────────────────────────────────────────────────────────────────────
#   DATE DTYPE                                                         
#  ────────────────────────────────────────────────────────────────────
df['date'] = df['date'].dt.date

#  ────────────────────────────────────────────────────────────────────
#   SAVE CLEANED DATASET
#  ────────────────────────────────────────────────────────────────────
df.to_parquet('./data/2-interim/stock-history-cleaned.parquet')
