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
    
    df = filename
    #print(df[:15])

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
            #print(df[df.columns[j]][:10]) 
               
        else:                       
            df[df.columns[j]] = df[df.columns[j]].astype('category', ordered=True).values
            df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.arange(len(feature),dtype=float))
            df[df.columns[j]].astype(int).values                        
            #print(df[df.columns[j]][:15])
        
    #print(df[:15])
    
    target = df['Happy']
    data = df.drop('Happy',axis=1)
       
    return {'data':data,'target':target}

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
    
    if (isClassified == True):
        groups = X.groupby(['Gender', 'Income'])

        for key,values in groups:  
            print(key)
            for j in range(values.shape[1]):
                name = values.columns[j]
                #print(name)
                temp = np.array(values[name])
                if (strategy == 'median'):
                    fill_value = np.nanmedian(temp, axis = 0)
                if (strategy == 'mean'):
                    fill_value = np.nanmean(temp, axis = 0)
                if (strategy == 'mode'):
                    fill_value = mode(temp,axis=0).mode[0]       

                if (name == 'UserID' or name == 'YOB' or name == 'votes'):
                        s =values.loc[:, name].fillna(fill_value)
                        values.loc[:, name] = s
                else:
                    if fill_value not in values.loc[:, name].cat.categories:                       
                        s = values.loc[:, name].cat.add_categories([fill_value])
                        values.loc[:, name]=s
                    s=values.loc[:, name].fillna(fill_value) 
                    values.loc[:, name] = s   
                
            

    if (isClassified == False):        
        for j in range(X.shape[1]):
            name = X.columns[j]            
            Col_j = np.array(X[name])
            
            if (strategy == 'median'):
                fill_value = np.nanmedian(Col_j, axis = 0)
            if (strategy == 'mean'):
                fill_value = np.nanmean(Col_j, axis = 0)
            if (strategy == 'mode'):
                fill_value = mode(Col_j,axis=0).mode[0]
 
            if (name == 'UserID' or name == 'YOB' or name == 'votes'):            
                X[name] = X[name].fillna(fill_value)   
            else:  
                if (strategy == 'mean'):
                    X[name] = X[name].cat.add_categories([fill_value])
                X[name] = X[name].fillna(fill_value)
                
    X_full = X  
    return X_full
    
def main():
    ## Read the raw data with pandas.read_csv()
    df = pd.read_csv('data/train.csv', index_col = None, na_values=["?"])
    df.replace('?', np.NaN)
    #print(len(df.dtypes))
    
    Dict = transform(df)
    X = Dict['data']
    y = Dict['target']
    
    print(X.shape, y.shape)

    #cats = pd.Categorical([1,0], categories=[1,0])
    #print(Dict['data']['Gender'].groupby(cats).mean())

    X_fill = fill_missing(X, 'mean', True)
    #print(X[340:347])

if __name__ == '__main__':
    main()
    
