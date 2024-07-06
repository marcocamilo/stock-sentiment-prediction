import pandas as pd
from src.modules.nlp import *
from tqdm import tqdm

df = pd.read_parquet('./data/1-raw/TSLA/tsla-news-master.parquet')

df['date'] = pd.to_datetime(df['date'])
print(f"Date span: {df.date.min()} – {df.date.max()}")

test = df.copy()
tqdm.pandas(desc = 'Text cleaning')
test[['title', 'summary']] = test[['title', 'summary']].progress_map(preprocessing, tokenize=False)
