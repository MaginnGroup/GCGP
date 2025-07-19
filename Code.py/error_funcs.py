import numpy as np

def MAE(data, prediction):
    mae = 1 / len(data) * sum(abs(data - prediction))
    return mae

def RMSE(data, prediction):
    mse = 1 / len(data) * sum((data - prediction) ** 2)
    return np.sqrt(mse)
