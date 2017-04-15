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
    #print(df[:15])

    df2 = np.array(df)
    #print(df.shape, df2.shape)
   
    for j in range(df2.shape[1]): 
        if j in [0,1,7,109]:
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
            df[df.columns[j]] = df[df.columns[j]].astype('category', categories=[np.nan,"under $25,000",
                "$25,001 - $50,000","$50,000 - $74,999","$75,000 - $100,000","$100,001 - $150,000",
                "over $150,000"], ordered=False).values     
            #print(df[df.columns[j]][50:70])      
            df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.insert(np.arange(len(feature),dtype=float),[0],[np.nan]))
            df[df.columns[j]].astype(int).values
            #print(df[df.columns[j]][50:70])   

        elif df.columns[j] == 'EducationLevel':            
            #print(df[df.columns[j]][:10]) 
            df[df.columns[j]] = df[df.columns[j]].astype('category', categories=[np.nan,"Current K-12",
                "High School Diploma","Current Undergraduate","Associate's Degree", 
                "Bachelor's Degree","Master's Degree","Doctoral Degree"], ordered=False).values          
            df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.insert(np.arange(len(feature),dtype=float),[0],[np.nan]))
            df[df.columns[j]].astype(int).values     
            #print(df[df.columns[j]][:10]) 
               
        else:                       
            df[df.columns[j]] = df[df.columns[j]].astype('category', ordered=True).values
            df[df.columns[j]] = df[df.columns[j]].cat.add_categories([np.nan])            
            #print(df[df.columns[j]].cat.categories.values)
            #print(df[df.columns[j]][:15])
            df[df.columns[j]] = df[df.columns[j]].cat.rename_categories(np.insert(np.arange(len(feature),dtype=float),[len(feature)],[np.nan]))
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
    
    if (isClassified == 0):       
        for j in range(3,4):#X.shape[1]):
            name = X.columns[j]
            Col_j = np.array(X[name])
                       
            #print(X[name][60:70]) 
            mean = np.nanmean(Col_j,axis=0)
            for i in range(len(Col_j)):
                if (np.isnan(Col_j[i])):                
                    #X[name][i] = mean
                    break
            #print(X[name][60:70])    
            
                  
    #return X_full
    
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

    fill_missing(X, 1, 0)

if __name__ == '__main__':
    main()
    
