import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

ticker = yf.Ticker('GOOG')
history = ticker.history(start='2022-01-01')

fig = go.Figure(
    data=[
        go.Candlestick(
            x = history.index,
            open = history['Open'],
            high = history['High'],
            low = history['Low'],
            close = history['Close']
        )
    ],
)

fig.update_layout(
    xaxis_rangeslider_visible=False,
    title = f'{ticker.ticker} January 2022 – present'
)

fig.show()

# tickers = ['TSLA', 'AAPL', 'GOOG']
# for ticker in tickers:
#
#     symbol = yf.Ticker(ticker)
#     history = symbol.history(start='2022-01-01')
#
#     file_symbol = ticker.lower()
#     history.to_parquet(f'./data/1-raw/{symbol.ticker}/{file_symbol}-history.parquet')
