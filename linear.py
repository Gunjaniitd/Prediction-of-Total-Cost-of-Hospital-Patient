import sys
import os

import numpy as pylib
from sklearn import linear_model 
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

import time

def linearRegression(dataFile, testFile, outputfile, weightfile):
	data = pylib.loadtxt(dataFile, dtype = str, delimiter = ",")

	data = data[ 1 : , 1 : ]
	data = data.astype('float64')

	X_dash = data[ : , : -1]
	n, m = X_dash.shape
	temp = pylib.ones((n, 1))
	X = pylib.concatenate((temp, X_dash), axis = 1)

	Y = data[ : , -1 : ]

	W = pylib.linalg.inv(X.T @ X)
	W = W @ X.T
	W = W @ Y

	pylib.savetxt(weightfile, W, delimiter = "\n")

	test = pylib.loadtxt(testFile, dtype = str, delimiter = ",")
	test = test[ 1 : , 1 : ]
	n, m = test.shape
	temp = pylib.ones((n, 1))
	test = pylib.concatenate((temp, test), axis = 1)
	test = test.astype('float64')

	predictedY = test @ W

	pylib.savetxt(outputfile, predictedY, delimiter = "\n")

def ridgeRegressionHelper(X, Y, lamda, X_test, Y_test):
	n, m = X.shape

	identity = pylib.ones((m, m))
	W = pylib.linalg.inv(lamda * identity + X.T @ X)
	W = W @ X.T
	W = W @ Y

	predictedY = X_test @ W

	error = pylib.sum(pylib.square(predictedY - Y_test)) / pylib.sum(pylib.square(Y))

	return error 

def ridgeRegression(dataFile, testFile, regularization, outputfile, weightfile, bestparameter):
	lamdaVector = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
	
	data = pylib.loadtxt(dataFile, dtype = str, delimiter = ",")

	data = data[ 1 : , 1 : ]
	data = data.astype('float64')

	X_dash = data[ : , : -1]
	n, m = X_dash.shape
	temp = pylib.ones((n, 1))
	X = pylib.concatenate((temp, X_dash), axis = 1)

	Y = data[ : , -1 : ]

	error = [0 for i in range(len(lamdaVector))]

	for k in range(10):
		size = n // 10

		X_k = pylib.concatenate((X[ : size * k, : ], X[size * (k + 1) : , : ]), axis = 0)
		Y_k = pylib.concatenate((Y[ : size * k, : ], Y[size * (k + 1) : , : ]), axis = 0)

		X_test = X[size * k : size * (k + 1), : ]
		Y_test = Y[size * k : size * (k + 1), : ]

		for lamda in range(len(lamdaVector)):
			error[lamda] += ridgeRegressionHelper(X_k, Y_k, lamdaVector[lamda], X_test, Y_test)


	index = pylib.argmin(error)
	best_lamda = lamdaVector[index]

	# print(best_lamda)

	pylib.savetxt(regularization, lamdaVector, delimiter = "\n")
	pylib.savetxt(bestparameter, [best_lamda], delimiter = "\n")

	identity = pylib.ones((m + 1, m + 1))
	W = pylib.linalg.inv(best_lamda * identity + X.T @ X)
	W = W @ X.T
	W = W @ Y

	pylib.savetxt(weightfile, W, delimiter = "\n")

	test = pylib.loadtxt(testFile, dtype = str, delimiter = ",")
	test = test[ 1 : , 1 : ]
	n, m = test.shape
	temp = pylib.ones((n, 1))
	test = pylib.concatenate((temp, test), axis = 1)
	test = test.astype('float64')

	predictedY = test @ W

	pylib.savetxt(outputfile, predictedY, delimiter = "\n")

dic_list = []

def meanEncoding(X, Y):
	global dic_list
	val = {}
	count = {}
	# print(len(X))
	for i in range(len(X)):
		if X[i] in val:
			val[X[i]] += Y[i]
			count[X[i]] += 1
		else:
			val[X[i]] = Y[i]
			count[X[i]] = 1

	for key in val:
		val[key] = val[key] / count[key]

	for i in range(len(X)):
		X[i] = val[X[i]]
	
	dic_list.append(val)
	return X

def meanEncode(X, index):
	global  dic_list
	dic = dic_list[index]

	for i in range(len(X)):
		if X[i] in dic:
			X[i] = dic[X[i]]

	return X

def lassoRegression(dataFile, testFile, outputfile):
	data = pylib.loadtxt(dataFile, dtype = str, delimiter = ",")

	X = data[1 : , 1 : -1].astype(float)
	Y = data[1 : , -1].astype(float)

	n, m = X.shape

	temp1 = X[ : , 1].reshape(n, 1)
	temp2 = X[ : , 3].reshape(n, 1)
	temp3 = X[ : , 10].reshape(n, 1)
	temp4 = X[ : , 12].reshape(n, 1)

	X = pylib.concatenate((X, temp1 + temp2), axis = 1)
	X = pylib.concatenate((X, temp1 + temp3), axis = 1)
	X = pylib.concatenate((X, temp1 + temp4), axis = 1)
	X = pylib.concatenate((X, temp2 + temp3), axis = 1)
	X = pylib.concatenate((X, temp2 + temp4), axis = 1)
	X = pylib.concatenate((X, temp3 + temp4), axis = 1)

	toRemove = [22, 20, 18, 16, 14, 4, 2]
	pylib.delete(X, toRemove, 1)

	inactiveFeatures = [0, 23, 24, 25, 30, 38, 41, 45, 47, 56, 60, 63, 64, 70, 72, 77, 83, 87, 88, 99, 
	112, 118, 120, 122, 129, 131, 136, 137, 141, 144, 146, 148, 150, 154, 155, 158, 160, 167, 170]

	n, m = X.shape

	for i in range(m):
		X[ : , i] = meanEncoding(X[ : , i], Y)

	temp = pylib.ones((n, 1))
	X = pylib.concatenate((temp, X), axis = 1)

	principalComponents = PCA(n_components = 17)
	polyCreator = PolynomialFeatures(2)

	X = principalComponents.fit_transform(X)
	X = polyCreator.fit_transform(X)

	pylib.delete(X, inactiveFeatures, 1)

	W = pylib.linalg.inv(X.T @ X)
	W = W @ X.T
	W = W @ Y

	test = pylib.loadtxt(testFile, dtype = str, delimiter = ",")
	test = test[ 1 : , 1 : ]
	test = test.astype('float64')

	n, m = test.shape

	temp1 = test[ : , 1].reshape(n, 1)
	temp2 = test[ : , 3].reshape(n, 1)
	temp3 = test[ : , 10].reshape(n, 1)
	temp4 = test[ : , 12].reshape(n, 1)

	test = pylib.concatenate((test, temp1 + temp2), axis = 1)
	test = pylib.concatenate((test, temp1 + temp3), axis = 1)
	test = pylib.concatenate((test, temp1 + temp4), axis = 1)
	test = pylib.concatenate((test, temp2 + temp3), axis = 1)
	test = pylib.concatenate((test, temp2 + temp4), axis = 1)
	test = pylib.concatenate((test, temp3 + temp4), axis = 1)

	pylib.delete(test, toRemove, 1)
	n,m = test.shape
	for i in range(m):
		test[ : , i] = meanEncode(test[ : , i], i)

	temp = pylib.ones((n, 1))
	test = pylib.concatenate((temp, test), axis = 1)

	test = principalComponents.transform(test)
	test = polyCreator.transform(test)

	pylib.delete(test, inactiveFeatures, 1)

	predictedY = test @ W

	pylib.savetxt(outputfile, predictedY, delimiter = "\n")

# lassoRegression("train_large.csv", "train.csv", "outputfile.txt")

if __name__ == '__main__':
	start = time.time()

	part = sys.argv[1]

	if (part == "a"):

		train = sys.argv[2]
		test = sys.argv[3]
		outputfile = sys.argv[4]
		weightfile = sys.argv[5]

		linearRegression(train, test, outputfile, weightfile)

	if (part == "b"):

		train = sys.argv[2]
		test = sys.argv[3]
		regularization = sys.argv[4]
		outputfile = sys.argv[5]
		weightfile = sys.argv[6]
		bestparameter = sys.argv[7]

		ridgeRegression(train, test, regularization, outputfile, weightfile, bestparameter)

	if (part == "c"):
		
		train = sys.argv[2]
		test = sys.argv[3]
		outputfile = sys.argv[4]

		lassoRegression(train, test, outputfile)

	end = time.time()
	# print("Time Taken: " + str(end - start))


