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
#   ADJUST FOR SPLITS                                                  
#  ────────────────────────────────────────────────────────────────────

#  ────────────────────────────────────────────────────────────────────
#   KEEP RELEVANT COLUMNS                                              
#  ────────────────────────────────────────────────────────────────────
df = df[['Date', 'Close', 'stock']]

#  ────────────────────────────────────────────────────────────────────
#   SAVE CLEANED DATASET
#  ────────────────────────────────────────────────────────────────────
df.to_parquet('./data/3-processed/stocks-history.parquet')
