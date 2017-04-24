from sklearn.preprocessing import LabelBinarizer
import numpy as np
#from sklearn import preprocessing

class NaiveBayes:
    def __init__(self):
        """ your code here """
        self._alpha = 1.0

    def fit(self, X, y):
        """ your code here """

        assert X.shape[0] == y.shape[0], "X and y must be consistent"

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self._y_label = np.sort(y.unique())
        #print(self._y_label)
        self._num_class = self._y_label.shape[0]

        samp_num = X.shape[0]
        #print(samp_num)
        self._feat_num = X.shape[1]

        self._log_pik = np.zeros((self._num_class,self._feat_num))
        #print(y.shape)
        X = np.array(X, dtype = float)
        y = np.array(y)
        y = np.reshape(y, (len(y), 1))
        #scaler = preprocessing.MinMaxScaler()
        #X = scaler.fit_transform(X)

        for i in range(samp_num):
            for j in range(self._num_class):
                if(y[i] == self._y_label[j]):
                    self._log_pik[j,:] = self._log_pik[j,:]+ X[i,:]


        for j in range(self._num_class):
        	self._log_pik[j,:] = (self._log_pik[j,:] + self._alpha) / (self._log_pik.sum(1)[j] + self._alpha * self._feat_num)

        self._log_pik = np.log(self._log_pik)

        #print(self._log_pik.shape)

        #priori calculation
        self._log_prio = []
        for i in self._y_label:
        	p = (y!=i).sum()/y.size
        	p = np.log(p)
        	self._log_prio.append(p)

        self._log_prio = np.array(self._log_prio)[:,np.newaxis]

        #print(self._log_prio.shape)
    	#print(y_log_prio[0])
    	#print(y_log_prio[1])
        return self

    def predict(self, X):
        """ your code here """
        X_T = np.transpose(np.array(X))

        #print("hhhh",X_T)
        #print(self._log_prio)
        #print(self._log_pik)

        y = np.zeros((X.shape[0],1))
        prediction = np.dot(self._log_pik,X_T) + self._log_prio
        prediction = np.transpose(prediction)

        y = np.argmax(prediction,axis=1)
        #print(y)
        return y

