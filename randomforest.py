import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from linear_regression import pre_process, prepare_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def main():
    dataset = pd.read_csv("training_dataset.csv")
    bare_df = pre_process(dataset)
    no_zeros = bare_df.fillna(value=0)
    # height_filter = age_filter[age_filter['Body Height [cm]']]
    # what rows were removed
    X = no_zeros.drop('Income in EUR', axis = 1)
    Y = no_zeros[['Income in EUR']]

    test_read = pd.read_csv("final_test.csv")
    test_df = pre_process(test_read)
    test_df = test_df.drop(['Income'], axis=1)
    test, X = test_df.align(X, join="inner", axis=1)

    #scores, predictions = rfr_model(X, Y.values.ravel())
    #print("Shape of predictons after forest: ", str(predictions.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
    regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print("Shape of x train: " + str(X_train.shape) + " Shape of y train: " + str(y_train.shape))
    regressor.fit(X_train, y_train.ravel())
    prediction = regressor.predict(X_test)
    model_r2 = r2_score(y_test, prediction)
    print("R2: {:.2}".format(model_r2))

    test = test.fillna(test.mean())
    prediction2 = regressor.predict(test)
    instance_array = test_df.get("Instance")
    predicted2 = np.stack((instance_array, prediction2.flatten()))
    predicted_final = pd.DataFrame(predicted2).T
    print("Shape of predicted final: " + str(predicted_final.shape))
    predicted_final.to_csv(r'predicted2.csv', header=['Instance','Income'])



if __name__ == '__main__':
    main()
