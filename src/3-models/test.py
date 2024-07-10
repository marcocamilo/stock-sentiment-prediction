import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense, Dropout

df = pd.read_parquet("./data/3-processed/stock-history-features.parquet")
df.insert(13, "close", df.pop("close"))
df.drop("date", axis=1, inplace=True)

tsla = df.query("stock == 'TSLA'").copy()
tsla.drop("stock", axis=1, inplace=True)
tsla = tsla.values

scaler = MinMaxScaler()
tsla = scaler.fit_transform(tsla)

def create_sequences(data, look_back=30):
    X, y = [], []

    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back])

    return np.array(X), np.array(y)

X, y = create_sequences(tsla)

train_factor = 0.8
split = int(len(X) * train_factor)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='rmsprop', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=30, batch_size=24, verbose=1)
