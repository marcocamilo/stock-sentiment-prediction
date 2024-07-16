import pandas as pd
from tqdm import tqdm
import spacy
from spacy_cleaner import Cleaner, processing

#  ────────────────────────────────────────────────────────────────────
#   JOIN STOCK DATASETS                                             
#  ────────────────────────────────────────────────────────────────────
stocks = ["TSLA", "AAPL", "GOOG"]

df = pd.concat(
    [pd.read_parquet(f'./data/1-raw/{stock}/{stock.lower()}-news-master.parquet').assign(stock=stock)
     for stock in tqdm(stocks)], ignore_index=True
)
print(df.shape)

#  ────────────────────────────────────────────────────────────────────
#   MISSING AND DULPLICATE VALUES CHECK                                
#  ────────────────────────────────────────────────────────────────────
print(df.isna().mean())
print(df.duplicated().mean())

display(df[df.duplicated()])
df = df.drop_duplicates()
print(df.duplicated().mean())

#  ────────────────────────────────────────────────────────────────────
#   DATE DTYPE AND FILTERING
#  ────────────────────────────────────────────────────────────────────
df['date'] = pd.to_datetime(df['date'], format="%Y%m%dT%H%M%S").dt.date
print(f"Date span: {df.date.min()} – {df.date.max()}")

# Keep rows with date before or equal to July 01 2024
filter = df['date'] <= pd.to_datetime("2024-07-01").date()
df = df[filter]
print(f"Date span: {df.date.min()} – {df.date.max()}")

display(df)

#  ────────────────────────────────────────────────────────────────────
#   TEXT CLEANING                                                      
#  ────────────────────────────────────────────────────────────────────
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def mutate_lemma_lower(tok) -> str:
    return tok.lemma_.lower()

cleaner = Cleaner(
    nlp,
    processing.remove_stopword_token,
    processing.remove_punctuation_token,
    mutate_lemma_lower,
)

df['title'] = cleaner.clean(df['title'], n_process=-1)
df['summary'] = cleaner.clean(df['summary'], n_process=-1)

# ────────────────────────────────────────────────────────────────────
#  SAVE CLEANED DATA
# ────────────────────────────────────────────────────────────────────
df.to_parquet('./data/2-interim/stock-news-cleaned.parquet')
