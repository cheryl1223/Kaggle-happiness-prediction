from preprocess import transform
from preprocess import fill_missing
from lr import LogisticRegression

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import linear_model

def main():
    # load training data
    filename_train = './data/train.csv'
    train_dataset = transform(filename_train)
    X = train_dataset['data']
    X = X.drop("UserID",1)
    y = train_dataset['target']

    # fill in missing data (optional)
    X_full = fill_missing(X, 'most_frequent', False)

    X_a, X_b, y_a, y_b = cross_validation.train_test_split(
        X_full, y, test_size=0.25, random_state=0)

    ### use the logistic regression
    print('Train the logistic regression classifier')
    """ your code here """
    lr_model = LogisticRegression()
    lr_model = lr_model.fit(X_a, y_a)
    lr_score = lr_model.score(X_b, y_b)
    #print("cross validation lr_score: ", lr_score)

    '''
    t_lr_model = linear_model.LogisticRegression()
    t_lr_model = t_lr_model.fit(X_a, y_a)
    t_lr_score = t_lr_model.score(X_b, y_b)
    print("cross validation sklearn_lr_score: ", t_lr_score)
    '''

    ### use the naive bayes
    print('Train the naive bayes classifier')
    # nb_model = ...

    ## use the svm
    print('Train the SVM classifier')
    # svm_model = ...

    ## use the random forest
    print('Train the random forest classifier')
    rf_model = RandomForestClassifier(n_estimators=15,random_state=12)
    rf_model = rf_model.fit(X_a, y_a)
    rf_score = rf_model.score(X_b, y_b)
    #print("cross validation rf_score: ", rf_score)

    ## get predictions
    filename_test = './data/test.csv'
    test_dataset = transform(filename_test)
    X_test_data = test_dataset['data']
    UserID = (X_test_data[X_test_data.columns[0]]).copy()

    X_test_data = X_test_data.drop("UserID",1)
    X_test = fill_missing(X_test_data, 'most_frequent', False)

    # lr_model predictions
    lr_pred = lr_model.predict(X_test)
    lr_Happy = pd.DataFrame({'Happy': lr_pred})
    predsLR = pd.concat([UserID, lr_Happy], axis=1)
    predsLR.to_csv('./predictions/lr_predictions.csv', index=False)

    #diff_score = t_lr_model.score(X_test, lr_pred)
    #print("lr_mode v.s. sklearn.lr_model on test data: ", diff_score)

    # rf_model predictions
    rf_pred = rf_model.predict(X_test)
    rf_Happy = pd.DataFrame({'Happy': rf_pred})
    predsLR = pd.concat([UserID, rf_Happy], axis=1)
    predsLR.to_csv('./predictions/rf_predictions.csv', index=False)
    #res = pd.read_csv('./predictions/rf_predictions.csv')
    #print(res)

if __name__ == '__main__':
    main()
