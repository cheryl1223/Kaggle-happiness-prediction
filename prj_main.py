from preprocess import transform
from preprocess import fill_missing
from lr import LogisticRegression
from naive_bayes import NaiveBayes
from sklearn.naive_bayes import MultinomialNB #GaussianNB

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.svm import SVC

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
        X_full, y, test_size=0.2, random_state=0)

    ### use the logistic regression
    print('Train the logistic regression classifier')
    """ your code here """
    clf_lr = LogisticRegression()
    clf_lr = clf_lr.fit(X_a, y_a)
    clf_lr_score = clf_lr.score(X_b, y_b)
    print("cross validation clf_lr_score: ", clf_lr_score)

    
    lr_model = linear_model.LogisticRegression()
    lr_model = lr_model.fit(X_a, y_a)
    lr_score = lr_model.score(X_b, y_b)
    print("cross validation sklearn_lr_score: ", lr_score)
    

    ### use the naive bayes
    print('Train the naive bayes classifier')
    clf_nb = NaiveBayes()
    clf_nb = clf_nb.fit(X_full,y)
    y_predict_me = clf_nb.predict(X_b)
    print("-------Wrong prediction of Multinomial NB is ", (y_b != y_predict_me).sum()/y_b.shape[0])

    nb_model = MultinomialNB()
    nb_model = nb_model.fit(X_full,y)
    y_predict = nb_model.predict(X_b)
    #print(y_predict.shape)
    print("-------Wrong prediction of Multinomial NB is ", (y_b != y_predict).sum()/y_b.shape[0])

    ## use the svm
    print('Train the SVM classifier')
    # svm_model = ...
    svm_model = SVC(kernel='rbf')
    svm_model = svm_model.fit(X_a,y_a)
    y_predict_svm = svm_model.predict(X_b)
    print("-------Wrong prediction of SVM SVC is ", (y_b != y_predict_svm).sum()/y_b.shape[0])

    ## use the random forest
    print('Train the random forest classifier')
    rf_model = RandomForestClassifier(n_estimators=15,random_state=12)
    rf_model = rf_model.fit(X_a, y_a)
    rf_score = rf_model.score(X_b, y_b)
    print("cross validation sklearn_rf_score: ", rf_score)

    ## get predictions
    filename_test = './data/test.csv'
    test_dataset = transform(filename_test)
    X_test_data = test_dataset['data']
    UserID = (X_test_data[X_test_data.columns[0]]).copy()

    X_test_data = X_test_data.drop("UserID",1)
    X_test = fill_missing(X_test_data, 'most_frequent', False)

    # clf_lr predictions
    lr_pred = clf_lr.predict(X_test)
    lr_Happy = pd.DataFrame({'Happy': lr_pred})
    predsLR = pd.concat([UserID, lr_Happy], axis=1)
    predsLR.to_csv('./predictions/lr_predictions.csv', index=False)

    #diff_score = t_lr_model.score(X_test, lr_pred)
    #print("lr_mode v.s. sklearn.lr_model on test data: ", diff_score)

    # clf_nb predictions
    nb_pred = clf_nb.predict(X_test)
    nb_Happy = pd.DataFrame({'Happy': nb_pred})
    predsLR = pd.concat([UserID, nb_Happy], axis=1)
    predsLR.to_csv('./predictions/nb_predictions.csv', index=False)

    # rf predictions
    rf_pred = rf_model.predict(X_test)
    rf_Happy = pd.DataFrame({'Happy': rf_pred})
    predsLR = pd.concat([UserID, rf_Happy], axis=1)
    predsLR.to_csv('./predictions/rf_predictions.csv', index=False)
    #res = pd.read_csv('./predictions/rf_predictions.csv')
    #print(res)

    # svm predictions
    svm_pred = svm_model.predict(X_test)
    svm_Happy = pd.DataFrame({'Happy': svm_pred})
    predsLR = pd.concat([UserID, svm_Happy], axis=1)
    predsLR.to_csv('./predictions/svm_predictions.csv', index=False)


if __name__ == '__main__':
    main()
