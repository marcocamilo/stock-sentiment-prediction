import pandas as pd
from tqdm import tqdm
import spacy

#  ────────────────────────────────────────────────────────────────────
#   CREATE MASTER DATASET                                              
#  ────────────────────────────────────────────────────────────────────
stocks = ["TSLA", "AAPL", "GOOG"]

df = pd.concat(
    [pd.read_parquet(f'./data/1-raw/{stock}/{stock.lower()}-news-master.parquet').assign(stock=stock)
     for stock in tqdm(stocks)], ignore_index=True
)
print(df.shape)

#  ────────────────────────────────────────────────────────────────────
#   DATE DTYPE AND FILTERING
#  ────────────────────────────────────────────────────────────────────
df['date'] = pd.to_datetime(df['date'], format="%Y%m%dT%H%M%S")
print(f"Date span: {df.date.min()} – {df.date.max()}")

# mask all rows with date july 04 2024 or later
mask = df['date'] < '2024-07-04'
df = df[mask]
print(f"Date span: {df.date.min()} – {df.date.max()}")

#  ────────────────────────────────────────────────────────────────────
#   MISSING AND DULPLICATE VALUES CHECK                                
#  ────────────────────────────────────────────────────────────────────
print(df.isna().mean())
print(df.duplicated().mean())

display(df[df.duplicated()])
df = df.drop_duplicates()
print(df.duplicated().mean())

#  ────────────────────────────────────────────────────────────────────
#   TEXT CLEANING                                                      
#  ────────────────────────────────────────────────────────────────────
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def preprocess_texts_multithread(texts):
    texts = list(texts) 
    processed_texts = []
    for doc in tqdm(nlp.pipe(texts, disable=["ner", "parser"], n_process=-1), total=len(texts)):
        processed_text = ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
        processed_texts.append(processed_text)
    return processed_texts

df['title'] = preprocess_texts_multithread(df['title'])
df['summary'] = preprocess_texts_multithread(df['summary'])

# ────────────────────────────────────────────────────────────────────
#  SAVE CLEANED DATA
# ────────────────────────────────────────────────────────────────────
df.to_parquet('./data/2-interim/stock-news-cleaned.parquet')
