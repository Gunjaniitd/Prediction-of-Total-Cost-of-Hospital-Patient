import sys
import os

import numpy as pylib
import pandas as pdlib

from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

def meanEncoding(X, Y):
	val = {}
	count = {}
	print(len(X))
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
	
	return X

def lassoRegression(dataFile):
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

	n, m = X.shape

	for i in range(m):
		X[ : , i] = meanEncoding(X[ : , i], Y)


	temp = pylib.ones((n, 1))
	X = pylib.concatenate((temp, X), axis = 1)

	principalComponents = PCA(n_components = 17)
	polyCreator = PolynomialFeatures(2)

	X = principalComponents.fit_transform(X)
	X = polyCreator.fit_transform(X)

	lamdaVector = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 3, 10, 100]

	score = [0 for i in range(len(lamdaVector))]
	size = n // 10

	for lamda in range(len(lamdaVector)):

		for k in range(10):

			model = linear_model.LassoLars(alpha = lamdaVector[lamda])

			X_k = pylib.concatenate((X[ : size * k, : ], X[size * (k + 1) : , : ]))
			Y_k = pylib.concatenate((Y[ : size * k], Y[size * (k + 1) :]))

			X_test = X[size * k : size * (k + 1), : ]
			Y_test = Y[size * k : size * (k + 1)]

			model.fit(X_k, Y_k)

			score[lamda] += model.score(X_test, Y_test)

			print(model.score(X_test, Y_test))

		print(lamda, lamdaVector[lamda], score[lamda])


	index = pylib.argmax(score)
	best_lamda = lamdaVector[index]
	best_score = score[index]

	toRemove = []
	model = linear_model.LassoLars(best_lamda)
	model.fit(X, Y)
	W = model.coef_

	for i in range(len(W)):
		if W[i] == 0:
			toRemove.append(i)

	print(toRemove)
  
	print(best_lamda, best_score / 10.0)

lassoRegression("train.csv")





