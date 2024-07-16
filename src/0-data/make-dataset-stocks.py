import pandas as pd
import yfinance as yf

tickers = ['TSLA', 'AAPL', 'GOOG']
for ticker in tickers:

    symbol = yf.Ticker(ticker)
    history = symbol.history(start='2022-01-01')

    file_symbol = ticker.lower()
    history.to_parquet(f'./data/1-raw/{symbol.ticker}/{file_symbol}-history.parquet')
