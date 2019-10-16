import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor
from linear_regression import pre_process
from pre_process import PreProcess

def format_to_csv(instance_array, prediction, filename):
    predicted = np.stack((instance_array, prediction.flatten()))
    predicted_final = pd.DataFrame(predicted).T
    print("Shape of predicted final: " + str(predicted_final.shape))
    predicted_final.to_csv(filename, header=['Instance','Income'])

def main():
    preprocess = PreProcess()
    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    X_train = preprocess.X_train
    y_train = preprocess.y_train
    X_test = preprocess.X_test
    y_test = preprocess.y_test
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    test_data = scaler.transform(preprocess.test)
    print("")
    best_model = ""
    max_testing_score = 0.0
    for learning_rate in lr_list:
        n_estimators = 800
        max_depth = 3
        gb_clf = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, min_samples_split=20, max_depth=max_depth, random_state=0)
        gb_clf.fit(X_train, y_train.ravel())
        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
        testing_score = gb_clf.score(X_test, y_test)
        print("Accuracy score (validation): {0:.3f}".format(testing_score))
        if (testing_score > max_testing_score):
            best_model = gb_clf
            max_testing_score = testing_score
            prediction = best_model.predict(test_data)
            filename = "predicted_gb_" + str(n_estimators)  + "_" + str(max_depth) + "_" + str(learning_rate) + ".csv"
            format_to_csv(preprocess.test_instance, prediction, filename)
            print("CSV created")

if __name__ == '__main__':
    main()
