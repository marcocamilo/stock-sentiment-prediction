import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

tsla = yf.Ticker('TSLA')
history = tsla.history(start='2022-01-01')

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
    title = 'TSLA 2022 – present'
)

fig.show()

history.to_parquet('./data/1-raw/TSLA/tsla-history.parquet')
