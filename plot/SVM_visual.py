from sklearn import svm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
import preprocess

filename = '../data/train.csv'
dataset = preprocess.transform(filename)
#dataset = preprocess.fill_missing(dataset,strategy = 'most_frequent',isClassified = False)

X = dataset['data']
y = dataset['target']

# drop NaN
total = (pd.concat([X, y], axis=1))
total = total.dropna()

#X = total.drop('Happy',1)
total = total[['Income','EducationLevel','Happy']].dropna()
total = preprocess.fill_missing(total,strategy = 'most_frequent',isClassified = False)

y = total['Happy']
X = total[['Income','EducationLevel']]
X = np.array(X)
y = np.array(y)

# train svm model
'''
svm_model = svm.SVC(kernel='rbf')
svm_model = svm_model.fit(X,y)
y_predict_svm = svm_model.predict(X)
'''

C = 1.0  # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf').fit(X, y)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
#lin_svc = svm.LinearSVC(C=C).fit(X, y)

#print(svc.score(X,y))
#print(rbf_svc.score(X,y))
#print(poly_svc.score(X,y))
#print(lin_svc.score(X,y))

h = 0.02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles ='SVM with rbf kernel'



Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Income')
plt.ylabel('Education Level')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles)
plt.savefig('SVM_visual')
plt.show()