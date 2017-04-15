import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn
seaborn.set()
from sklearn.preprocessing import StandardScaler

def transform(filename):
    """ preprocess the training data"""
    """ your code here """
    
    df = filename
    #print(df[:5])

    df2 = np.array(df)
    #print(df.shape, df2.shape)
   
    for j in range(df2.shape[1]): 

        dfj = set(df2[:,j])        
        #print(dfj)
        feature = []
        for x in dfj:
            if type(x) == float and np.isnan(x) == True:
                continue
            else:
                feature.append(x)   
        #print(df[df.columns[5]][:10])
        df[df.columns[j]] = df[df.columns[j]].astype('category',ordered=True).values            
        df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.arange(len(feature)))
        df[df.columns[j]].astype(int).values                        
        #print(df[df.columns[5]][:10])
        
    #print(df[:5])
    
    
    target = df['Happy']  
    data = df.drop('Happy',axis = 1)
    #print(df['Party'][19:100])
    return {'data':data,'target':target}
'''
def fill_missing(X, strategy, isClassified):
    """
     @X: input matrix with missing data filled by nan
     @strategy: string, 'median', 'mean', 'most_frequent'
     @isclassfied: boolean value, if isclassfied == true, then you need build a
     decision tree to classify users into different classes and use the
     median/mean/mode values of different classes to fill in the missing data;
     otherwise, just take the median/mean/most_frequent values of input data to
     fill in the missing data
    """
    """ your code here """
    X_full = X
    if isClassified == true:
        groups = X['data'].groupby(["Gender","Income"])
        #print (df.describe())
        for key, values in groups:
            if (strategy == "median"):
                for column in range(values.shape[1]):
                    median = values[column].median()
                X_full['column']=df['column'].fillna(value)

            if (strategy == "mean"):

            if (strategy == "most_frequent"):

            else:
                print ("invalid strategy")
    else:

    
    
    return X_full
'''    
def main():
    ## Read the raw data with pandas.read_csv()
    df = pd.read_csv('data/train.csv', index_col = None, na_values=["?"])
    df.replace('?', np.NaN)
    #print(len(df.dtypes))
    
    Dict = transform(df)
    print (Dict['data'].shape)
    #print(Dict['data']["Gender"].value_counts())
    X_full = Dict['data']
    groups = Dict['data'].groupby(["Gender","Income"]) 
    #fill_missing(Dict['data'], strategy, isClassified)

    #groups = X['data'].groupby(["Gender","Income"])

    n = 0
    for key, values in groups:
        print (key)
        for column in values:
            n = n+values[column].shape[0]
            df = values[column].to_frame()
            #print(df.mean())
        
    print (n)
if __name__ == '__main__':
    main()
    
