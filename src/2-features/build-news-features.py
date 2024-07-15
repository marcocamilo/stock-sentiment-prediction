import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#  ────────────────────────────────────────────────────────────────────
#   IMPORT DATA AND SET DEVICE
#  ────────────────────────────────────────────────────────────────────
df = pd.read_parquet("./data/2-interim/stock-news-cleaned.parquet")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using {device}.")

#  ────────────────────────────────────────────────────────────────────
#   TOKENIZATION AND SENTIMENT ANALYSIS
#  ────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(
    device
)

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def collate_fn(batch):
    return tokenizer(batch, padding=True, truncation=True, return_tensors="pt")


def sentiment_analyzer(texts, batch_size=64):
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_scores = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        scores = outputs.logits.softmax(dim=-1).detach().cpu().numpy()
        all_scores.extend(scores)

    all_scores = np.vstack(all_scores)

    return all_scores


titles = df["title"].to_list()
title_sentiments = sentiment_analyzer(titles)

summaries = df["summary"].to_list()
summary_sentiments = sentiment_analyzer(summaries)

sentiment_columns = [
    "title_pos",
    "title_neg",
    "title_neu",
    "summary_pos",
    "summary_neg",
    "summary_neu",
]
df[sentiment_columns[:3]] = title_sentiments
df[sentiment_columns[3:]] = summary_sentiments
display(df)

sentiment_aggs = df.groupby(['stock', 'date'])[sentiment_columns].mean().reset_index()
display(sentiment_aggs)

#  ────────────────────────────────────────────────────────────────────
#   SAVE DATA
#  ────────────────────────────────────────────────────────────────────
df.to_parquet("news-sentiment-features.parquet")
sentiment_aggs.to_parquet("news-sentiment-aggs.parquet")
