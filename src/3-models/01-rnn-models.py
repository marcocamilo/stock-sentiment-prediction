import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
import keras_tuner as kt

figsize = (16,4)
plt.rcParams['figure.figsize'] = figsize
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#  ────────────────────────────────────────────────────────────────────
#   IMPORT DATA
#  ────────────────────────────────────────────────────────────────────
df = pd.read_parquet("./data/3-processed/stock-history-features.parquet")
display(df)

#  ────────────────────────────────────────────────────────────────────
#   DATA PREPARATION                                                   
#  ────────────────────────────────────────────────────────────────────
class StockData:
    def __init__(self, stock, data, look_back=30, test_size=0.2):
        self.stock = stock
        self.data = data.query(f"stock == '{stock}'").copy()
        self.data.drop("stock", axis=1, inplace=True)
        self.look_back = look_back
        self.test_size = test_size
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
        self.X = self.data.values
        self.y = self.data['close'].values.reshape(-1, 1)
        
    def create_sequences(self, data):
        input_data = []
        target_data = []
        for i in range(len(data[0]) - self.look_back):
            input_seq = data[0][i:i+self.look_back]
            input_data.append(input_seq)

            target_value = data[1][i+self.look_back]
            target_data.append(target_value)
        return np.array(input_data), np.array(target_data)
        
    def scale_split_sequence(self):
        X_scaled = self.x_scaler.fit_transform(self.X)
        y_scaled = self.y_scaler.fit_transform(self.y)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, 
                                                            test_size=self.test_size, 
                                                            shuffle=False)
        
        X_train, y_train = self.create_sequences([X_train, y_train])
        X_test, y_test = self.create_sequences([X_test, y_test])
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform(self, data, scaler='y'):
        scaler_dict = {
            'x': self.x_scaler,
            'y': self.y_scaler
        }
        data_inv = scaler_dict[scaler].inverse_transform(data)
        
        return data_inv

TSLA = StockData("TSLA", df)
AAPL = StockData("AAPL", df)
GOOG = StockData("GOOG", df)

#  ────────────────────────────────────────────────────────────────────
#   HYPERMODEL                                                         
#  ────────────────────────────────────────────────────────────────────
class GRUHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        
        # Hyperparameter settings
        num_gru_layers = hp.Int("num_gru_layers", 1, 3)
        gru_units = [hp.Int(f"gru_units_{i}", min_value=25, max_value=200, step=25) for i in range(num_gru_layers)]
        optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        # Add GRU layers
        for i in range(num_gru_layers):
            model.add(
                GRU(
                    units=gru_units[i],
                    return_sequences=True if i < num_gru_layers - 1 else False,
                    input_shape=(self.look_back, self.input_shape) if i == 0 else None
                )
            )
        
        # Add Output layer
        model.add(Dense(1))

        # Optimizer
        opt = {
            'adam': tf.keras.optimizers.Adam,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'sgd': tf.keras.optimizers.SGD
        }

        # Compile model
        model.compile(
            optimizer=opt[optimizer_choice](learning_rate=learning_rate),
            loss='mse'
        )
        
        return model

#  ────────────────────────────────────────────────────────────────────
#   MODEL BUILDING                                                     
#  ────────────────────────────────────────────────────────────────────
class StockModel:
    def __init__(self, stock_data, hypermodel=None):
        self.stock_data = stock_data
        self.hypermodel = hypermodel if hypermodel else GRUHyperModel()

        self.X_train, self.X_test, self.y_train, self.y_test = self.stock_data.scale_split_sequence()
        self.input_shape = self.X_train.shape[2]
        self.hypermodel.look_back = self.stock_data.look_back
        self.hypermodel.input_shape = self.input_shape

        self.best_model = None

    def train_best_model(self, max_trials=30, epochs=20, validation_split=0.2, monitor='val_loss', patience=5, min_delta=0):
        tuner = kt.RandomSearch(
            self.hypermodel,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=2,
            directory='dir',
            project_name=f'{self.stock_data.stock}'
        )

        stop_early = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)

        tuner.search(self.X_train, self.y_train, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])

        self.best_model = tuner.get_best_models(num_models=1)[0]
        
        return self.best_model

    def evaluate_model(self):
        predictions = self.best_model.predict(self.X_test)
        pred = self.stock_data.inverse_transform(predictions, scaler='y')
        true = self.stock_data.inverse_transform(self.y_test, scaler='y')

        rmse = np.sqrt(mean_squared_error(true, pred))
        print(f"The root mean squared error is {rmse:.2f} USD for {self.stock_data.stock} model.")

        plt.plot(true, label='True')
        plt.plot(pred, label='Predicted')
        plt.title(f"{self.stock_data.stock} Model Performance")
        plt.legend()
        plt.show()

#  ────────────────────────────────────────────────────────────────────
#   TSLA                                                               
#  ────────────────────────────────────────────────────────────────────
tsla_model = StockModel(TSLA, GRUHyperModel())
best_tsla_model = tsla_model.train_best_model()
best_tsla_model.summary()

tsla_model.evaluate_model()

#  ────────────────────────────────────────────────────────────────────
#   AAPL                                                               
#  ────────────────────────────────────────────────────────────────────
aapl_model = StockModel(AAPL, GRUHyperModel())
best_aapl_model = aapl_model.train_best_model()
best_aapl_model.summary()

aapl_model.evaluate_model()

#  ────────────────────────────────────────────────────────────────────
#   GOOG                                                               
#  ────────────────────────────────────────────────────────────────────
goog_model = StockModel(GOOG, GRUHyperModel())
best_goog_model = goog_model.train_best_model()
best_goog_model.summary()

goog_model.evaluate_model()

