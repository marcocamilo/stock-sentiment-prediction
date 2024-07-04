import pandas as pd
from tqdm import tqdm
import requests
from datetime import datetime, timedelta

def fetch_articles(ticker, start_date_str, api_key, limit=1000):
    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
    news = []

    # Convert start_date_str to datetime object
    start_date = datetime.strptime(start_date_str, '%Y%m%dT%H%M')

    current_date = datetime.now()
    total_days = (current_date - start_date).days

    with tqdm(total=total_days, desc="Fetching Articles") as pbar:
        while True:
            params = dict(
                tickers=ticker,
                time_from=start_date.strftime('%Y%m%dT%H%M'),
                limit=limit,
                sort='EARLIEST',
                apikey=api_key,
            )
            
            try:
                r = requests.get(url, params=params)
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                return None

            if data.get('items') == '0':
                print("No more articles to extract!")
                return None
                
            if 'Information' in data:
                print(data['Information'])
                print(f"{len(news)} articles extracted up to {start_date}!")
                break
            
            for item in data['feed']:
                news.append([item['time_published'], item['title'], item['summary']])
            
            # Convert to DataFrame
            df = pd.DataFrame(news, columns=['date', 'title', 'summary'])
            
            # Check the last date in the DataFrame and update start_date
            last_date_str = dataframe['date'].iloc[-1]
            last_date = df.strptime(last_date_str, '%Y%m%dT%H%M%S')

            # Update progress bar
            days_progress = (last_date - start_date).days
            pbar.update(days_progress)

            # Update start_date to continue fetching
            start_date = last_date + timedelta(minutes=1)
            
            # Ensure the progress bar does not exceed total days
            if start_date >= current_date:
                print("Current date reached!")
                break

    return df

ticker = 'TSLA'
start_date_str = '20220101T0000'
api_key = 'demo'

df = fetch_articles(ticker, start_date_str, api_key)
display(df)

# df.to_parquet('../data/1-raw/tsla-news.parquet')
