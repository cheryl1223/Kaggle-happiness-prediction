import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from math import isnan
from sklearn import datasets, linear_model, preprocessing
import seaborn
seaborn.set()
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer   
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation

import warnings
warnings.filterwarnings("ignore")

def transform(filename):
    df = pd.read_csv(filename, index_col = None, na_values=["?"])
    df.replace('?', np.NaN)
    target = []
    if 'Happy' in df.columns:
        target = df['Happy']
        df = df.drop('Happy', axis = 1) 
    UserID = df[['UserID']]
    numeric_col = df[['YOB', 'votes']]
    categorical_col = df[['Income','HouseholdStatus','EducationLevel','Party']]
    binary_col = df.drop(['UserID','YOB','Income','HouseholdStatus',\
                        'EducationLevel','Party','votes'],axis = 1)

    #one_hot = pd.get_dummies(categorical_col)
    for i in categorical_col:
        if i == 'Income':
            categorical_col['Income']=categorical_col['Income'].map({"under $25,000":0,
                "$25,001 - $50,000":1,"$50,000 - $74,999":2,"$75,000 - $100,000":3,"$100,001 - $150,000":4,\
                "over $150,000":5})
        if i == 'HouseholdStatus':
            categorical_col['HouseholdStatus']=categorical_col['HouseholdStatus'].map({"Single (no kids)":0,\
                "Married (no kids)":1,"Domestic Partners (no kids)":2,"Domestic Partners (w/kids)":3,\
                "Single (w/kids)":4, "Married (w/kids)":5})
        if i == 'EducationLevel':
            categorical_col['EducationLevel'] = categorical_col['EducationLevel'].map({"Current K-12":0,\
                "High School Diploma":1,"Current Undergraduate":2,"Associate's Degree":3,\
                "Bachelor's Degree":4,"Master's Degree":5,"Doctoral Degree":6})
        if i == 'Party':
            categorical_col['Party'] = categorical_col['Party'].map({"Libertarian":0,"Democrat":1,\
                "Other":2,"Republican":3, "Independent":4})

    for i in binary_col: 
        if i == 'Gender':
            binary_col[i] = binary_col[i].map({'Male': 0, 'Female': 1})

        types = binary_col[i].unique()
        types = [x for x in types if str(x) != 'nan']
        binary_col[i] = binary_col[i].map({types[0]: 0, types[1]: 1})

    #binary_col = fill_missing(binary_col,'mean', False)
    #numeric_col = fill_missing(numeric_col,'mean', False)
    #categorical_col = fill_missing(categorical_col,'mean', False)
    data = pd.concat([UserID, numeric_col,categorical_col,binary_col], axis=1)
    return {'data':data,'target':target}

def fill_missing(X, strategy, isClassified):
    imp = Imputer(missing_values='NaN', strategy=strategy, axis=1)
    if (isClassified == False): 
        for column in X:
            X[column] = np.transpose(imp.fit_transform([X[column]]))
    if (isClassified == True):

        X['Gender'] = np.transpose(imp.fit_transform([X['Gender']]))
        X['Income'] = np.transpose(imp.fit_transform([X['Income']]))
        groups = X.groupby(['Gender', 'Income'])

        for key,values in groups:
            for column in values:
                name = values[column]
                temp = np.array(values[column])     
                fill_value = np.transpose(imp.fit_transform(temp))
                INDEX = values[column].index.values
                X.loc[INDEX,column] = fill_value
        #X = X.dropna()
        #X = X.sort('UserID')
    return X  

def normalize(X):
    scaler = StandardScaler() 
    X['YOB'] = np.transpose(scaler.fit_transform(X['YOB']))
    X['votes'] = np.transpose(scaler.fit_transform(X['votes']))
    return X

def main():
    # load training data
    print("####################################################") 
    print("Data preprocessing...")   
    filename = 'data/train.csv'
    Dict = transform(filename)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        Dict['data'],Dict['target'], test_size=0.2, random_state=0)
    
    train = pd.concat([X_train,y_train],axis = 1)
    print ("Cross validate on 10% of the whole dataset, training data shape: ", train.shape)
    thresh = 0.2*train.shape[1]
    train = train.dropna(thresh = thresh)
    train = train.dropna(subset = ['Income','HouseholdStatus','EducationLevel'])

    train = fill_missing(train,strategy = 'most_frequent',isClassified = False)

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


    X_test = fill_missing(X_test, strategy = 'most_frequent',isClassified = False)
    UserID = X_test['UserID']

    categorical_col = X_test[['Income','HouseholdStatus','EducationLevel','Party']]
    numeric_col = X_test[['YOB', 'votes']]
    binary_col = X_test.drop(['UserID','YOB','Income','HouseholdStatus',\
                        'EducationLevel','Party','votes'],axis = 1)
    one_hot = encoder.fit_transform(categorical_col).toarray()
    normalized = normalize(numeric_col)

    X_test = np.hstack((normalized,one_hot,binary_col))
    X_a = X
    X_b = X_test
    y_a = y
    y_b = y_test


if __name__ == '__main__':
    main()
    
