import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor
from linear_regression import pre_process
from pre_process import PreProcess
from sklearn.neighbors import KNeighborsRegressor

from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

def main():
    preprocess = PreProcess()
    X_train = preprocess.X_train
    y_train = preprocess.y_train
    X_test = preprocess.X_test
    y_test = preprocess.y_test
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    test_data = scaler.transform(preprocess.test)
    rmse_val = []
    for k in range(30):
        k = k+1
        model = neighbors.KNeighborsRegressor(n_neighbors = k)
        model.fit(X_train, y_train)  #fit the model
        pred=model.predict(X_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(error) #store rmse values
        print('RMSE value for k= ' , k , 'is:', error)

    curve = pd.DataFrame(rmse_val) #elbow curve
    curve.plot()

if __name__ == '__main__':
    main()
