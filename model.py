import matplotlib.pyplot as plt
import numpy as np
import cPickle
from time import time,sleep
from sklearn.cross_validation import train_test_split
import sklearn.linear_model as lm
from sklearn import metrics
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from os import system,getpid

np.set_printoptions(suppress=True)
system("taskset -p 0xff %d" % getpid())

def one_hot_vect(a,nCats=None):
	if nCats is None:
		aNew = np.zeros((len(a),np.max(a)+1))
	else:
		aNew = np.zeros((len(a),nCats))
	for i in xrange(len(a)):
		aNew[i,a[i]] = 1
	return aNew

def load(path,n_chars=5,n_examples='all',test_size=0.2,seed=42):
	texts = cPickle.load(open('data/data.pkl'))
	text = ' '.join(texts)
	text = text[:n_examples+n_chars]
	ords = np.array([ord(char) for char in text])
	ords = ords-ords.min()
	data = one_hot_vect(ords)
	print data.shape
	X = []
	for i in range(n_chars,len(data)):
		X.append(data[i-n_chars:i].flatten())
	# X = csr_matrix(np.array(X))
	X = np.array(X)
	Y = data[n_chars:]
	Y = np.argmax(Y,axis=1)
	# print Y.min()
	# print Y.max()
	Y = np.array(Y)
	# if n_examples is not 'all':
	# 	indexes = np.arange(0,X.shape[0],1)
	# 	np.random.seed(seed)
	# 	mask = np.random.choice(indexes,size=n_examples,replace=False)
	# 	X = X[mask]
	# 	Y = Y[mask]
	trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_size, random_state = seed)
	return trainX,trainY,testX,testY

n_examples = 100000
trainX,trainY,testX,testY = load('data/data.pkl',n_chars=12,n_examples=n_examples)
print trainX.shape,trainY.shape,testX.shape,testY.shape

t = time()
# model = lm.Ridge()
model = RandomForestClassifier(n_estimators=12,n_jobs=2,verbose=2)
# model = lm.LogisticRegression()
# model = LinearSVC()
model.fit(trainX,trainY)
pred = model.predict(testX)
# pred = np.argmax(pred,axis=1)
# print pred
# print np.argmax(testY,axis=1)
print metrics.accuracy_score(testY,pred),time()-t
# print metrics.auc_score(testY.flatten(),pred.flatten()),time()-t