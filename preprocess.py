import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn import datasets, linear_model, preprocessing
import seaborn
seaborn.set()
from sklearn.preprocessing import StandardScaler


def transform(filename):
    """ preprocess the training data"""
    """ your code here """
    
    df = pd.read_csv(filename, index_col = None, na_values=["?"])
    df.replace('?', np.NaN)
    #print(df[:15])
    #df = df.drop("UserID",1)
    #print(list(df))
    #print(df["Happy"].unique())
    df2 = np.array(df)
    #print(df.shape, df2.shape)
   
    for j in range(df2.shape[1]):
        if (df.columns[j] == 'UserID' or df.columns[j] == 'YOB' or
            df.columns[j] == 'votes' or df.columns[j] == 'Happy'):
            continue
        
        dfj = set(df2[:,j])        
        #print(dfj)
        feature = []
        for x in dfj:
            if type(x) == float and np.isnan(x) == True:
                continue
            else:
                feature.append(x)   
        #print(df[df.columns[5]][:10])
        if df.columns[j] == 'Income':
            #print(df[df.columns[j]][:10]) 
            df[df.columns[j]] = df[df.columns[j]].astype('category', categories=["under $25,000",
                "$25,001 - $50,000","$50,000 - $74,999","$75,000 - $100,000","$100,001 - $150,000",
                "over $150,000"], ordered=False).values     
            #print(df[df.columns[j]][50:70])      
            df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.arange(len(feature),dtype=float))
            df[df.columns[j]].astype(int).values
            #print(df[df.columns[j]][50:70])   

        elif df.columns[j] == 'EducationLevel':            
            #print(df[df.columns[j]][:10]) 
            df[df.columns[j]] = df[df.columns[j]].astype('category', categories=["Current K-12",
                "High School Diploma","Current Undergraduate","Associate's Degree", 
                "Bachelor's Degree","Master's Degree","Doctoral Degree"], ordered=False).values          
            df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.arange(len(feature),dtype=float))
            df[df.columns[j]].astype(int).values

        elif df.columns[j] == 'Party':            
            #print(df[df.columns[j]][:10]) 
            df[df.columns[j]] = df[df.columns[j]].astype('category', categories=["Lbertarian","Democrat",
                "Other","Republican", "Independent"], ordered=False).values          
            df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.arange(len(feature),dtype=float))
            df[df.columns[j]].astype(int).values     
            #print(df[df.columns[j]][:10])

        elif df.columns[j] == 'HouseholdStatus':            
            #print(df[df.columns[j]][:10]) 
            df[df.columns[j]] = df[df.columns[j]].astype('category', categories=["Single (no kids)","Married (no kids)",
                "Domestic Partners (no kids)","Domestic Partners (w/kids)", "Single (w/kids)", "Married (w/kids)"], ordered=False).values          
            df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.arange(len(feature),dtype=float))
            df[df.columns[j]].astype(int).values 
               
        else:                       
            df[df.columns[j]] = df[df.columns[j]].astype('category', ordered=True).values
            df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.arange(len(feature),dtype=float))
            df[df.columns[j]].astype(int).values                        
            #print(df[df.columns[j]][:15])
        
    #print(df[:15])
    #print(df[341:346])
    data = df
    target = []
    for j in range(df2.shape[1]): 
        if df.columns[j] == 'Happy':
            target = df['Happy']
            data = df.drop('Happy',axis=1)
            break
       
    return {'data':data,'target':target}

def fill_missing(Y, strategy, isClassified):
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
    X = Y.copy()
    if (isClassified == True):
        groups = X.groupby(['Gender', 'Income'])
        #print(groups.dtypes)
        
        for key,k_values in groups:
            #print(key)            
            #print(k_values.shape)
            for j in range(k_values.shape[1]):
                name = k_values.columns[j]
                #print(values[name][:5])
                temp = np.array(k_values[name])
                if (strategy == 'median'):
                    fill_value = np.nanmedian(temp, axis = 0)
                if (strategy == 'mean'):
                    fill_value = np.nanmean(temp, axis = 0)
                if (strategy == 'mode'):
                    fill_value = mode(temp,axis=0).mode[0]       
                
                INDEX = k_values.index.values

                if (name == 'UserID' or name == 'YOB' or name == 'votes'):
                    X.loc[INDEX,name] = X.loc[INDEX,name].fillna(fill_value)
                else:
                    if fill_value not in X.loc[:, name].cat.categories:                       
                        X.loc[:, name] = X.loc[:, name].cat.add_categories([fill_value])
                    X.loc[INDEX,name] = X.loc[INDEX,name].fillna(fill_value)
                    
        X_full = X.dropna()
        #print(X_full.shape)
        return X_full
    
    if (isClassified == False):        
        for j in range(X.shape[1]):
            name = X.columns[j]            
            Col_j = np.array(X[name])
            
            if (strategy == 'median'):
                fill_value = np.nanmedian(Col_j, axis = 0)
            if (strategy == 'mean'):
                fill_value = np.nanmean(Col_j, axis = 0)
            if (strategy == 'most_frequent'):
                fill_value = mode(Col_j,axis=0).mode[0]
 
            if (name == 'UserID' or name == 'YOB' or name == 'votes'):            
                X[name] = X[name].fillna(fill_value)   
            else:  
                if (strategy == 'mean'):
                    X[name] = X[name].cat.add_categories([fill_value])
                X[name] = X[name].fillna(fill_value)                
        X_full = X 
        return X_full
    
#def main():
#    ## Read the raw data with pandas.read_csv()
#    #df = pd.read_csv('data/train.csv', index_col = None, na_values=["?"])
#    #df.replace('?', np.NaN)
#    #print(len(df.dtypes))
#    filename = 'data/train.csv'
#    Dict = transform(filename)
#    x = Dict['data']
#    y = Dict['target']
#    
#    #print(X.shape, y.shape)
#    
#    X_fill = fill_missing(x, 'mean', True)
#    #print(x[341:342])
#    #print(X_fill[310:311])
#
#if __name__ == '__main__':
#    main()
    
