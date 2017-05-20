from preprocess import *
from lr import LogisticRegression
from naive_bayes import NaiveBayes
from sklearn.naive_bayes import BernoulliNB,GaussianNB
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# Feel free to modify the parameters!
# Default parameters were tuned accoring to accuracy on cross validation test set
test_size = 0.1
random_state = 12
strategy = "most_frequent"
isClassified = False
nan_thresh = 0.8
kernel = 'rbf'
n_estimators = 100


def main():
    
    # load training data
    print("####################################################") 
    print("Data preprocessing...")   
    filename = 'data/train.csv'
    Dict = transform(filename)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        Dict['data'],Dict['target'], test_size=test_size, random_state=random_state)
    
    train = pd.concat([X_train,y_train],axis = 1)
    print ("Cross validate on 10% of the whole dataset, training data shape: ", train.shape)
    thresh = nan_thresh*train.shape[1]
    train = train.dropna(thresh = thresh)
    #train = train.dropna(subset = ['Income','HouseholdStatus','EducationLevel'])

    train = fill_missing(train,strategy,isClassified)

    y = train['Happy']
    df = train.drop('Happy',1)
    numeric_col = df[['YOB', 'votes']]
    #numeric_col = fill_missing(numeric_col,'mean',False)
    normalized = pd.DataFrame(normalize(numeric_col))
    categorical_col = df[['Income','HouseholdStatus','EducationLevel','Party']]
    #categorical_col = fill_missing(categorical_col,'median',False)
    binary_col = df.drop(['UserID','YOB','Income','HouseholdStatus',\
                        'EducationLevel','Party','votes'],axis = 1)

    encoder = OneHotEncoder()
    #one_hot = pd.get_dummies(categorical_col)
    one_hot = pd.DataFrame(encoder.fit_transform(categorical_col).toarray())

    #X = pd.concat([normalized,one_hot,binary_col],axis = 1)

    X = np.hstack((normalized,one_hot,binary_col))
    #X = np.hstack((one_hot,binary_col))


    X_test = fill_missing(X_test, strategy,isClassified)
    UserID = X_test['UserID']

    categorical_col = X_test[['Income','HouseholdStatus','EducationLevel','Party']]
    numeric_col = X_test[['YOB', 'votes']]
    binary_col = X_test.drop(['UserID','YOB','Income','HouseholdStatus',\
                        'EducationLevel','Party','votes'],axis = 1)
    one_hot = encoder.fit_transform(categorical_col).toarray()
    normalized = normalize(numeric_col)
    #X = np.hstack((one_hot,binary_col))

    X_test = np.hstack((normalized,one_hot,binary_col))
    X_a = X
    X_b = X_test
    y_a = y
    y_b = y_test
    print("Training data shape (After drop NaN and one hot encoding): ", X_a.shape)
    print("Cross validation data shape (after one hot encoding): ", X_b.shape)
    #X_a, X_b, y_a, y_b = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)
    print("####################################################")    
    ### use the logistic regression
    print('Train the logistic regression classifier...')
    start = time.time()
    clf_lr = LogisticRegression()
    clf_lr = clf_lr.fit(X_a, y_a)
    clf_lr_predict = clf_lr.predict(X_b)
    end = time.time()
    print("Accuracy of logistic regression is ", (y_b == clf_lr_predict).sum()/y_b.shape[0])
    print("Time of of logistic regression is %.4fs."%(end-start))

    start = time.time()
    lr_model = linear_model.LogisticRegression()
    lr_model = lr_model.fit(X_a, y_a)
    lr_model_predict = lr_model.predict(X_b)
    print("Scikit learn n_iter_: ", lr_model.n_iter_)
    end = time.time()
    print("Accuracy of scikit-learn logistic Regression is ", (y_b == lr_model_predict).sum()/y_b.shape[0])
    print("Time of of scikit-learn logistic regression is %.4fs."%(end-start))
    print(" ")
   
    ### use the naive bayes
    print('Train the naive bayes classifier...')
    start = time.time()
    clf_nb = NaiveBayes()
    clf_nb = clf_nb.fit(X_a,y_a)
    clf_nb_predict = clf_nb.predict(X_b)
    end = time.time()
    print("Accuracy of naive bayes  is ", (y_b == clf_nb_predict).sum()/y_b.shape[0])
    print("Time of of naive bayes is %.4fs."%(end-start))


    start = time.time()
    nb_model = BernoulliNB()
    nb_model = nb_model.fit(X_a,y_a)
    nb_model_predict = nb_model.predict(X_b)
    end = time.time()
    print("Accuracy of scikit-learn Bernoulli NB is ", (y_b == nb_model_predict).sum()/y_b.shape[0])
    print("Time of of scikit-learn Bernoulli NB is %.4fs."%(end-start))
    print(" ")
    
    ## use the svm
    print('Train the SVM classifier...')
    start = time.time()
    svm_model = SVC(kernel = kernel)
    svm_model = svm_model.fit(X_a,y_a)
    svm_model_predict = svm_model.predict(X_b)
    end = time.time()

    print("Accuracy of scikit-learn SVM is ", (y_b == svm_model_predict).sum()/y_b.shape[0])
    print("Time of of scikit-learn SVM is %.4fs."%(end-start))
    print(" ")
   
    ## use the random forest
    print('Train the random forest classifier...')
    start = time.time()
    rf_model = RandomForestClassifier(n_estimators=n_estimators,random_state=random_state)
    rf_model = rf_model.fit(X_a, y_a)
    rf_model_predict = rf_model.predict(X_b)
    end = time.time()

    print("Accuracy of scikit-learn RF is ", (y_b == rf_model_predict).sum()/y_b.shape[0])
    print("Time of of scikit-learn RF is %.4fs."%(end-start))
    print(" ")

    print("####################################################") 
    print("Start predicting test dataset...")    
    filename_test = './data/test.csv'
    Dict = transform(filename_test)
    X_test_data = Dict['data']

    test = fill_missing(X_test_data, strategy,isClassified)
    UserID = test['UserID'].astype(int)

    categorical_col = test[['Income','HouseholdStatus','EducationLevel','Party']]
    numeric_col = test[['YOB', 'votes']]
    binary_col = test.drop(['UserID','YOB','Income','HouseholdStatus',\
                        'EducationLevel','Party','votes'],axis = 1)
    one_hot = encoder.fit_transform(categorical_col).toarray()
    normalized = normalize(numeric_col)
    #X_test = np.hstack((one_hot,binary_col))

    X_test = np.hstack((normalized,one_hot,binary_col))
    print("Test data shape (after one hot encoding): ", X_test.shape)  

    # clf_lr predictions
    lr_pred = clf_lr.predict(X_test)
    lr_Happy = pd.DataFrame({'Happy': lr_pred})
    predsLR = pd.concat([UserID, lr_Happy], axis=1)
    predsLR.to_csv('./predictions/lr_predictions.csv', index=False)


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

    # svm predictions
    svm_pred = svm_model.predict(X_test)
    svm_Happy = pd.DataFrame({'Happy': svm_pred})
    predsLR = pd.concat([UserID, svm_Happy], axis=1)
    predsLR.to_csv('./predictions/svm_predictions.csv', index=False)
    print("Prediction saved.")   

if __name__ == '__main__':
    main()
