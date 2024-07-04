import pandas as pd

df = pd.read_parquet('./data/1-raw/TSLA/tsla-news-master.parquet')

df['date'] = pd.to_datetime(df['date'])
print(f"Date span: {df.date.min()} – {df.date.max()}")
