from preprocess import transform
from preprocess import fill_missing

def main():
    # load training data
    filename_train = './data/train.csv'
    train_dataset = transform(filename_train)
    X = train_dataset['data']
    y = train_dataset['target']

    # fill in missing data (optional)
    X_full = fill_missing(X, 'most_frequent', False)

    ### use the logistic regression
    print('Train the logistic regression classifier')
    """ your code here """
    # lr_model = ...

    ### use the naive bayes
    print('Train the naive bayes classifier')
    """ your code here """
    # nb_model = ...

    ## use the svm
    print('Train the SVM classifier')
    """ your code here """
    # svm_model = ...

    ## use the random forest
    print('Train the random forest classifier')
    """ your code here """
    # rf_model = ...

    ## get predictions
    """ your code here """

if __name__ == '__main__':
    main()
