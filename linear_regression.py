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
from sklearn.preprocessing import PolynomialFeatures

def pre_process(dataset2):
    dataset2['Gender'] = dataset2.get("Gender").fillna('')
    dataset2['Country'] = dataset2.get("Country").fillna('')
    dataset2['University Degree'] = dataset2.get("University Degree").fillna('')
    dataset2['Profession'] = dataset2.get("Profession").fillna('')

    dataset2['Gender'] = pd.Categorical(dataset2['Gender'])
    gender_dummies = pd.get_dummies(dataset2['Gender'], prefix = 'gender')

    dataset2['Country'] = pd.Categorical(dataset2['Country'])
    country_dummies = pd.get_dummies(dataset2['Country'], prefix = 'country')

    dataset2['Profession'] = pd.Categorical(dataset2['Profession'])
    profession_dummies = pd.get_dummies(dataset2['Profession'], prefix = 'profession')

    dataset2['University Degree'] = pd.Categorical(dataset2['University Degree'])
    uni_dummies = pd.get_dummies(dataset2['University Degree'], prefix = 'University Degree')

    bare_df = dataset2.drop(['Gender', 'Country', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color'], axis=1)
    dataset2 = pd.concat([bare_df, profession_dummies], axis=1)
    dataset2 = pd.concat([dataset2, gender_dummies], axis=1)
    bare_df = pd.concat([bare_df, country_dummies], axis=1)
    bare_df = pd.concat([bare_df, uni_dummies], axis=1)
    return bare_df

def prepare_model(bare_df, test_df):
    # bare_df = bare_df[bare_df['Age']>13]
    # height_filter = age_filter[age_filter['Body Height [cm]']]
    # what rows were removed
    X = no_zeros.drop('Income in EUR', axis = 1)
    Y = no_zeros[['Income in EUR']]
    train, X = test_df.align(X, join="inner", axis=1)
    print("Shape of X after align: ", str(X.shape))
    X_ = np.nan_to_num(X)
    Y_ = np.nan_to_num(Y)

    polynomial_features= PolynomialFeatures(degree=2)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.25, random_state=12)
    x_poly = polynomial_features.fit_transform(X_train)
    regression_model = LinearRegression()
    regression_model.fit(x_poly, y_train)
    predicted = regression_model.predict(x_poly)
    return regression_model, predicted, X_test, train

def main():
    dataset = pd.read_csv("training_dataset.csv")
    bare_df = pre_process(dataset)
    print("Columns in bare_df: ")
    print(bare_df.columns)

    test_read = pd.read_csv("final_test.csv")
    test_df = pre_process(test_read)
    test_df = test_df.drop(['Income'], axis=1)

    regression_model, predicted, y_test, test_df = prepare_model(bare_df, test_df)
    instance_array = test_df.get("Instance")

    test_df_ = np.nan_to_num(test_df)
    predicted2 = regression_model.predict(test_df_)
    predicted2 = np.stack((instance_array, predicted2.flatten()))
    print("Shape of predicted2: " + str(predicted2.shape))
    predicted_final = pd.DataFrame(predicted2).T
    print("Shape of predicted final: " + str(predicted_final.shape))
    predicted_final.to_csv(r'predicted2.csv', header=['Instance','Income'])
    # np.savetxt("predicted2.csv", predicted2, delimiter=",", header="Results")

    # model_mse = mean_squared_error(y_test, predicted)
    # # calculate the mean absolute error
    # model_mae = mean_absolute_error(y_test, predicted)
    # # calulcate the root mean squared error
    # model_rmse = math.sqrt(model_mse)
    # # display the output
    # print("MSE {:f}".format(model_mse))
    # print("MAE {:f}".format(model_mae))
    # print("RMSE {:f}".format(model_rmse))

    #model_r2 = r2_score(y_test, predicted2)
    #print("R2: {:.2}".format(model_r2))






if __name__ == '__main__':
    main()
