import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from time import time
import pickle

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 500

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, 64, 64))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X)
X_test_pca = pca.transform(Xtest)
print("done in %0.3fs" % (time() - t0))

rounds = 5

classifiers = [
    ("SGD", SGDClassifier(n_jobs=-1)),
    ("ASGD", SGDClassifier(average=True, n_jobs=-1)),
    ("Perceptron", Perceptron()),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge', n_jobs=-1,
                                                         C=1.0)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge', n_jobs=-1,
                                                          C=1.0)),
    ("SAG", LogisticRegression(solver='sag', n_jobs=-1, tol=1e-1, C=1.e4 / len(X)))
]

xx = [x for x in range(len(classifiers))]
yy = []
for name, clf in classifiers:
    print("training %s" % name)
    
    yy_ = []
    for r in range(rounds):
        clf.fit(X_train_pca, y)
        y_pred = clf.predict(X_test_pca)
        yy_.append(np.mean(y_pred == ytest))
    yy.append(np.mean(yy_))

# Compute confusion matrix
cnf_matrix = confusion_matrix(ytest, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True, classes=['vide', 'poing', 'main'],
                      title='Confusion matrix, without normalization')
plt.show()

plt.scatter(xx,yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Classifier")
plt.ylabel("Test Success Rate")
plt.show()


#print(clf.score(X, y))
#print(clf.score(Xtest, ytest))
