# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

#Début
def recup(dossier):
	X_train = pickle.load(open('./dataTrain/Xtrain.dump', 'rb'))
	X_test = pickle.load(open('./dataTrain/Xtest.dump', 'rb'))
	y_test = pickle.load(open('./dataTrain/Ytest.dump', 'rb'))
	y_train = pickle.load(open('./dataTrain/Ytrain.dump', 'rb'))
	X_testClass = pickle.load(open('./dataTrain/XtestClass.dump', 'rb'))
	y_testClass = pickle.load(open('./dataTrain/YtestClass.dump', 'rb'))
	return X_train, y_train, X_test, y_test, X_testClass, y_testClass

X_train, y_train, X_test, y_test, X_testClass, y_testClass = recup('dataTrain')
print(len(X_train), len(X_test))

y = []
X = []
for x in range(len(y_train)):
	X.append(X_train[x].flatten())
	y.append(np.argmax(y_train[x]))

ytest = []
Xtest = []
for x in range(len(y_test)):
	Xtest.append(X_test[x].flatten())
	ytest.append(np.argmax(y_test[x]))

print(X_train[0])
print('')
print(y[0])

print("recuperation done")

# Plot the PCA spectrum
pca.fit(X)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# Prediction
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X, y)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()