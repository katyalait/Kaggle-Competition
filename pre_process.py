import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats

class PreProcess:
    def __init__(self):
        self.X, self.Y, self.X_train, self.y_train, self.X_test, self.y_test, self.test, self.test_instance = self.clean()

    def encoding(self, dataset2):
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

    def clean(self):
        dataset = pd.read_csv("training_dataset.csv")
        bare_df = self.encoding(dataset)
        no_zeros = bare_df.fillna(value=0)
        # height_filter = age_filter[age_filter['Body Height [cm]']]
        # what rows were removed
        print("Shape of no zeros: " + str(no_zeros.shape))
        X = no_zeros.drop('Income in EUR', axis = 1)
        Y = no_zeros[['Income in EUR']]
        test_read = pd.read_csv("final_test.csv")
        test_df = self.encoding(test_read)
        test_df = test_df.drop(['Income'], axis=1)
        test, X = test_df.align(X, join="inner", axis=1)
        X_train, y_train, X_test, y_test = self.get_training_sets(X, Y)
        test_instance = test.get('Instance')
        test = test.fillna(value=0)
        return X, Y, X_train, y_train, X_test, y_test, test, test_instance

    def get_training_sets(self, X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=12)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print("Shape of x train: " + str(X_train.shape) + " Shape of y train: " + str(y_train.shape))
        return X_train, y_train, X_test, y_test
