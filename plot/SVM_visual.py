from sklearn import svm
import pandas as pd
import numpy as np

import sys
sys.path.append('../')
import preprocess

filename = '../data/train.csv'
dataset = preprocess.transform(filename)
X = dataset['data']
y = dataset['target']

# drop NaN
total = (pd.concat([X, y], axis=1)).dropna()
X = total.drop('Happy',1)
y = total['Happy']

# train svm model
'''
svm_model = svm.SVC(kernel='rbf')
svm_model = svm_model.fit(X,y)
y_predict_svm = svm_model.predict(X)
'''

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

'''
data = X.copy()
for j in data.columns:
	if (j == 'Income' or j == 'EducationLevel'):
		continue
	else:
		data[j] = 0
'''

data = X[['Income','EducationLevel']]

data = np.array(data)
print(data.shape)

h = 0.02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

print(X.shape, xx.shape, y.shape, yy.shape)
# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()