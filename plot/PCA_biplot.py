import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import sys
sys.path.append('../')

from preprocess import *
## import data
#filename = '.././data/train.csv'
#df = pd.read_csv(filename)
my_csv = '.././data/train.csv' ## path to your dataset

dat = pd.read_csv(my_csv, index_col=0)
dat = transform(my_csv)
data = dat['data']
target = dat['target']
dat = pd.concat([data, target], axis=1)
dat = dat.drop("UserID",1)

#dat = dat.drop("YOB",1)
#dat = fill_missing(dat,'most_frequent', False)
# if no row or column titles in your csv, pass 'header=None' into read_csv
# and delete 'index_col=0' -- but your biplot will be clearer with row/col names
dat = dat.dropna()
'''
categorical_col = dat[['Income','HouseholdStatus','EducationLevel','Party']]
encoder = OneHotEncoder()
categorical_col = pd.DataFrame(encoder.fit_transform(categorical_col).toarray())
dat = dat.drop(['Income','HouseholdStatus','EducationLevel','Party'],axis = 1)


dat = dat.join(categorical_col)

dat = dat.dropna()
'''
#dat = normalize(dat)
df = dat["Happy"]
dat = dat.drop('Happy',1)
dat = dat.join(df)
#name = dat.columns
#scaler = MinMaxScaler()
#dat = pd.DataFrame(data = scaler.fit_transform(dat), columns= name)

## perform PCA

n = len(dat.columns)

pca = PCA(n_components = n)
# defaults number of PCs to number of columns in imported data (ie number of
# features), but can be set to any integer less than or equal to that value

pca.fit(dat)



## project data into PC space

# 0,1 denote PC1 and PC2; change values for other PCs
xvector = pca.components_[0] # see 'prcomp(my_data)$rotation' in R
yvector = pca.components_[1]



dat_happy = dat.loc[dat['Happy'] == 1]
dat_unhappy = dat.loc[dat['Happy'] == 0]
xs_happy = pca.transform(dat_happy)[:,0] # see 'prcomp(my_data)$x' in R
ys_happy = pca.transform(dat_happy)[:,1]

xs_unhappy = pca.transform(dat_unhappy)[:,0] # see 'prcomp(my_data)$x' in R
ys_unhappy = pca.transform(dat_unhappy)[:,1]


xs = pca.transform(dat)[:,0]
ys = pca.transform(dat)[:,1]


## visualize projections
    
## Note: scale values for arrows and text are a bit inelegant as of now,
##       so feel free to play around with them

for i in range(0,len(xvector)-1):
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='b', width=0.0005, head_width=0.0025)
    plt.text(xvector[i]*max(xs), yvector[i]*max(ys),
             list(dat.columns.values)[i], color='g')

plt.arrow(0, 0, xvector[len(xvector)-1]*max(xs)*5, yvector[len(xvector)-1]*max(ys)*5,
              color='r', width=0.0005, head_width=0.0025)
plt.text(xvector[len(xvector)-1]*max(xs)*5, yvector[len(xvector)-1]*max(ys)*5,
             list(dat.columns.values)[len(xvector)-1], color='r')

for i in range(len(xs_happy)):
# circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs_happy[i], ys_happy[i], 'yo')

for i in range(len(xs_unhappy)):
    plt.plot(xs_unhappy[i], ys_unhappy[i], 'ro')

    #plt.text(xs[i], ys[i], list(dat.index)[i], color='b',fontsize = 10)
plt.savefig('PCA_biplot')
plt.show()