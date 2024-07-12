import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f} USD.".format(rmse))
