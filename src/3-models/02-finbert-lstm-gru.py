import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv1D, LSTM, GRU, Dense, Attention, Add, LayerNormalization, Dropout, GlobalAveragePooling1D
import keras_tuner as kt

figsize = (16,4)
plt.rcParams['figure.figsize'] = figsize
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#  ────────────────────────────────────────────────────────────────────
#   IMPORT DATA                                                        
#  ────────────────────────────────────────────────────────────────────
sentiments = pd.read_parquet('/kaggle/input/news-sentiment-aggs/news-sentiment-aggs.parquet')
stocks_full = pd.read_parquet('/kaggle/input/stock-sentiment-features/stock-history-features-full.parquet')
# stocks = pd.read_parquet('/kaggle/input/stock-sentiment-features/stock-history-features.parquet')

display(sentiments.sample(5))
display(stocks.sample(5))
display(stocks_full.sample(5))

#  ────────────────────────────────────────────────────────────────────
#   FILTER AND CONCATENATE DATASETS                                    
#  ────────────────────────────────────────────────────────────────────
print(
f"""Range of dates:
    Stocks: {stocks.date.min()} - {stocks.date.max()}
    Sentiments: {sentiments.date.min()} - {sentiments.date.max()}"""
)

stock_dates = set(stocks.date)
sent_dates = set(sentiments.date)

print(
f"""Number of dates:
    Stocks: {len(stock_dates)}
    Sentiments: {len(sent_dates)}"""
)

diff = pd.to_datetime(list(sent_dates.difference(stock_dates)))
diff.day_of_week.value_counts()

# df = sentiments.merge(stocks, how='inner', on=['date', 'stock'])
df = sentiments.merge(stocks_full, how='inner', on=['date', 'stock'])
df = df.set_index('date')
df.sample(5)


#  ────────────────────────────────────────────────────────────────────
#   DATA PREPARATION                                                   
#  ────────────────────────────────────────────────────────────────────
class StockData:
    def __init__(self, stock, data, look_back=30, test_size=0.2):
        self.stock = stock
        self.data = data.query(f"stock == '{stock}'").copy()
        self.index = self.data.index
        self.data.drop("stock", axis=1, inplace=True)
        self.look_back = look_back
        self.test_size = test_size
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
        self.X = self.data.values
        self.y = self.data['close'].values.reshape(-1, 1)
        
        self.X_scaled = self.x_scaler.fit_transform(self.X)
        self.y_scaled = self.y_scaler.fit_transform(self.y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_sequence()

    def create_sequences(self, data):
        input_data = []
        target_data = []
        for i in range(len(data[0]) - self.look_back):
            input_seq = data[0][i:i+self.look_back]
            input_data.append(input_seq)

            target_value = data[1][i+self.look_back]
            target_data.append(target_value)
        return np.array(input_data), np.array(target_data)
        
    def split_sequence(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y_scaled, 
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
    
    def plot(self):
        data = self.data
        split = int(len(data) * (1 - self.test_size))

        plt.plot(data.iloc[:split, -1], label='Training data')
        plt.plot(data.iloc[split:, -1], label='Test data')
        plt.xlabel("Date")
        plt.title("Stock Price")
        plt.legend()
        plt.show()
        plt.show()

TSLA = StockData("TSLA", df)
AAPL = StockData("AAPL", df)
GOOG = StockData("GOOG", df)

#  ────────────────────────────────────────────────────────────────────
#   HYPERMODELS                                                        
#  ────────────────────────────────────────────────────────────────────
class GRUHyperModel(kt.HyperModel):
    def __init__(self):
        self.model = "GRU"
    def build(self, hp):
        model = Sequential()
        
        # Hyperparameter settings
        num_gru_layers = hp.Int("num_gru_layers", 1, 3)
        gru_units = [hp.Int(f"gru_units_{i}", min_value=25, max_value=200, step=25) for i in range(num_gru_layers)]
        optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        # Input Layer
        model.add(Input(shape=(self.look_back, self.input_shape)))
        
        # Add GRU layers
        for i in range(num_gru_layers):
            model.add(
                GRU(
                    units=gru_units[i],
                    return_sequences=True if i < num_gru_layers - 1 else False
                )
            )
        
        # Add Output layer
        model.add(Dense(1))

        # Optimizer
        opt = {
            'adam': tf.keras.optimizers.Adam,
            'rmsprop': tf.keras.optimizers.RMSprop,
        }

        # Compile model
        model.compile(
            optimizer=opt[optimizer_choice](learning_rate=learning_rate),
            loss='mse'
        )
        
        return model

class LSTMHyperModel(kt.HyperModel):
    def __init__(self):
        self.model = "LSTM"
        
    def build(self, hp):
        model = Sequential()
        
        # Hyperparameter settings
        num_lstm_layers = hp.Int("num_lstm_layers", 1, 3)
        lstm_units = [hp.Int(f"lstm_units_{i}", min_value=25, max_value=200, step=25) for i in range(num_lstm_layers)]
        optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        # Input Layer
        model.add(Input(shape=(self.look_back, self.input_shape)))
        
        # Add LSTM layers
        for i in range(num_lstm_layers):
            model.add(
                LSTM(
                    units=lstm_units[i],
                    return_sequences=True if i < num_lstm_layers - 1 else False
                )
            )
        
        # Add Output layer
        model.add(Dense(1))

        # Optimizer
        opt = {
            'adam': tf.keras.optimizers.Adam,
            'rmsprop': tf.keras.optimizers.RMSprop,
        }

        # Compile model
        model.compile(
            optimizer=opt[optimizer_choice](learning_rate=learning_rate),
            loss='mse'
        )
        
        return model

class AttentionCNNLSTMHyperModel(kt.HyperModel):
    def __init__(self):
        self.model = "Attention-CNNLSTM"
        
    def build(self, hp):
        model = Sequential()
        
        # Hyperparameter settings
        kernel_size = hp.Int('kernel_size', min_value=3, max_value=5, step=1)
        cnn_filters = hp.Int('cnn_filters', min_value=32, max_value=256, step=32)
        lstm_units = hp.Int('lstm_units', min_value=50, max_value=200, step=50)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
        
        # Input Layer
        model.add(Input(shape=(self.look_back, self.input_shape)))
        
        # CNN Layers
        model.add(Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(Conv1D(filters=cnn_filters * 2, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(Conv1D(filters=cnn_filters * 4, kernel_size=kernel_size, activation='relu', padding='same'))
        
        # Attention Mechanism
        query = Dense(cnn_filters * 4)(model.layers[-1].output)
        key = Dense(cnn_filters * 4)(model.layers[-1].output)
        value = Dense(cnn_filters * 4)(model.layers[-1].output)
        attention_output = Attention()([query, key, value])
        
        # Add and Normalize
        add_output = Add()([model.layers[-1].output, attention_output])
        norm_output = LayerNormalization()(add_output)
        
        # LSTM Decoder
        model.add(LSTM(lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Output Layer
        model.add(Dense(1))
        
        # Optimizer
        opt = {
            'adam': tf.keras.optimizers.Adam,
            'rmsprop': tf.keras.optimizers.RMSprop,
        }

        # Compile model
        model.compile(
            optimizer=opt[optimizer_choice](learning_rate=learning_rate),
            loss='mse'
        )
        
        return model

#  ────────────────────────────────────────────────────────────────────
#   STOCK MODEL                                                        
#  ────────────────────────────────────────────────────────────────────
class StockModel:
    def __init__(self, stock_data, hypermodel=None, max_trials=20, epochs=20, directory='models'):
        self.stock_data = stock_data
        self.directory = directory
        self.max_trials = max_trials
        self.epochs = epochs

        self.X_train = self.stock_data.X_train
        self.X_test = self.stock_data.X_test
        self.y_train = self.stock_data.y_train
        self.y_test = self.stock_data.y_test
        self.input_shape = self.X_train.shape[2]
        
        self.hypermodel = hypermodel if hypermodel else GRUHyperModel()
        self.hypermodel.look_back = self.stock_data.look_back
        self.hypermodel.input_shape = self.input_shape

        self.best_model = None
        self.best_hyperparameters = None
        
        self.rmse = None
        self.accuracy = None

    def train_best_model(self, monitor='val_loss', patience=5, min_delta=0):
        tuner = kt.RandomSearch(
            self.hypermodel,
            objective='val_loss',
            max_trials=self.max_trials,
            executions_per_trial=2,
            directory=f'{self.directory}',
            project_name=f'{self.stock_data.stock}/{self.hypermodel.model}'
        )

        stop_early = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)

        tuner.search(self.X_train, self.y_train, epochs=self.epochs, validation_split=0.2, callbacks=[stop_early])
        
        self.best_model = tuner.get_best_models(num_models=1)[0]
        self.best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0].values
        
        return self.best_model

    def evaluate_model(self):
        y_true = self.y_test
        y_pred = self.best_model.predict(self.X_test)
        pred = self.stock_data.inverse_transform(y_pred, scaler='y')
        true = self.stock_data.inverse_transform(y_true, scaler='y')

        self.rmse = np.sqrt(mean_squared_error(true, pred))
        print(f"Root Mean Squared Error: {self.rmse:.2f} USD")
        
        self.accuracy = (1 - np.mean(np.abs((true - pred) / true))) * 100
        print(f"Model Accuracy: {self.accuracy:.2f}%")

        data = self.stock_data.data
        split = -pred.size

        plt.plot(data.index[split:], true, label='Test data')
        plt.plot(data.index[split:], pred, label='Predicted data', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"{self.stock_data.stock} Model Predictions")
        plt.legend()
        plt.show()
        
        plt.plot(data.index[:split], data.iloc[:split]['close'], label='Training data', color='lightgray')
        plt.plot(data.index[split:], true, label='Test data')
        plt.plot(data.index[split:], pred, label='Predicted data')
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"{self.stock_data.stock} Model Performance")
        plt.legend()
        plt.show()

#  ────────────────────────────────────────────────────────────────────
#   TSLA                                                               
#  ────────────────────────────────────────────────────────────────────
TSLA.plot()

tsla_lstm = StockModel(TSLA, LSTMHyperModel())
best_tsla_lstm = tsla_lstm.train_best_model()
best_tsla_lstm.summary()

tsla_gru = StockModel(TSLA, GRUHyperModel())
best_tsla_gru = tsla_gru.train_best_model()
best_tsla_gru.summary()

tsla_acnnlstm = StockModel(TSLA, AttentionCNNLSTMHyperModel())
best_tsla_acnnlstm = tsla_acnnlstm.train_best_model()
best_tsla_acnnlstm.summary()

tsla_lstm.evaluate_model()
tsla_gru.evaluate_model()
tsla_acnnlstm.evaluate_model()

#  ────────────────────────────────────────────────────────────────────
#   AAPL                                                               
#  ────────────────────────────────────────────────────────────────────
AAPL.plot()

aapl_lstm = StockModel(AAPL, LSTMHyperModel())
best_aapl_lstm = aapl_lstm.train_best_model()
best_aapl_lstm.summary()

aapl_gru = StockModel(AAPL, GRUHyperModel())
best_aapl_gru = aapl_gru.train_best_model()
best_aapl_gru.summary()

aapl_acnnlstm = StockModel(AAPL, AttentionCNNLSTMHyperModel())
best_aapl_acnnlstm = aapl_acnnlstm.train_best_model()
best_aapl_acnnlstm.summary()

aapl_lstm.evaluate_model()
aapl_gru.evaluate_model()
aapl_acnnlstm.evaluate_model()

#  ────────────────────────────────────────────────────────────────────
#   GOOG                                                               
#  ────────────────────────────────────────────────────────────────────
GOOG.plot()

goog_lstm = StockModel(GOOG, LSTMHyperModel())
best_goog_lstm = goog_lstm.train_best_model()
best_goog_lstm.summary()

goog_gru = StockModel(GOOG, GRUHyperModel())
best_goog_gru = goog_gru.train_best_model()
best_goog_gru.summary()

goog_acnnlstm = StockModel(GOOG, AttentionCNNLSTMHyperModel())
best_goog_acnnlstm = goog_acnnlstm.train_best_model()
best_goog_acnnlstm.summary()

goog_lstm.evaluate_model()
goog_gru.evaluate_model()
goog_acnnlstm.evaluate_model()

#  ────────────────────────────────────────────────────────────────────
#   EVALUATION                                                         
#  ────────────────────────────────────────────────────────────────────
models = dict(
    tsla_lstm = tsla_lstm, 
    tsla_gru = tsla_gru,
    aapl_lstm = aapl_lstm,
    aapl_gru = aapl_gru,
    goog_lstm = goog_lstm, 
    goog_gru = goog_gru,
    tsla_acnnlstm = tsla_acnnlstm,
    aapl_acnnlstm = aapl_acnnlstm,
    goog_acnnlstm = goog_acnnlstm,
)

results = pd.DataFrame()

for k, v in models.items():
    stock, model = k.split('_')
    params = v.best_hyperparameters
    accuracy = v.accuracy
    rmse = v.rmse
    
    data = pd.DataFrame(params, index=[len(results)])
    data['Model'] = model.upper()
    data['Stock'] = stock.upper()
    data['Accuracy'] = accuracy
    data['RMSE'] = rmse
    
    results = pd.concat([results, data], ignore_index=True)
    
results['num_layers'] = results['num_lstm_layers'].fillna(results['num_gru_layers'])
results['units_0'] = results['lstm_units_0'].fillna(results['gru_units_0'])
results['units_1'] = results['lstm_units_1'].fillna(results['gru_units_1'])
results['units_2'] = results['lstm_units_2'].fillna(results['gru_units_2'])
results.drop(['num_lstm_layers', 'num_gru_layers', 'lstm_units_0', 'gru_units_0',
             'lstm_units_1', 'gru_units_1', 'lstm_units_2', 'gru_units_2'], axis=1, inplace=True)

results = results[['Stock', 'Model', 'Accuracy', 'RMSE', 'num_layers', 'units_0', 'units_1', 'units_2', 
                    'optimizer', 'learning_rate', 'cnn_filters', 'kernel_size']]

display(results)

results.to_csv('finbert-rnn-results.csv')
