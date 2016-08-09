import re

from geotext.geotext import GeoText
#import geograpy

class LogisticRegression(object):
    """LogisticRegression classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Cost in every epoch.

    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            y_val = self.activation(X)
            errors = (y - y_val)
            neg_grad = X.T.dot(errors)
            self.w_[1:] += self.eta * neg_grad
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append(self._logit_cost(y, self.activation(X)))
        return self

    def _logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))
        return logit

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ Activate the logistic neuron"""
        z = self.net_input(X)
        return self._sigmoid(z)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
          Class 1 probability : float

        """
        return self.activation(X)

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        class : int
            Predicted class label.

        """
        # equivalent to np.where(self.activation(X) >= 0.5, 1, 0)
        return np.where(self.net_input(X) >= 0.0, 1, 0)


import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
df.tail()

import numpy as np

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 1, 0)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


lr = LogisticRegression(n_iter=500, eta=0.2).fit(X_std, y)
plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Logistic Regression - Learning rate 0.01')

plt.tight_layout()
plt.show()

plot_decision_regions(X_std, y, classifier=lr)
plt.title('Logistic Regression - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


def feature_Cap(word):
    if word[0].isupper():
        return 1
    else:
        return 0


def feature_prefix(word):
    return word[0:2]


def feature_suffix(word):
    return word[-2:]

def suf_fixes(words):
    suffixes = set()
    for line in words:
        if len(line.split())>1:
            suffixes.add(feature_suffix(line.split()[0]))
    suffixes = list(suffixes)
    return suffixes


def pre_fixes(words):
    prefixes = set()
    for line in words:
        if len(line.split())>1:
            prefixes.add(feature_prefix(line.split()[0]))
    prefixes = list(prefixes)
    return prefixes

from nltk.corpus import names


def feature_nameList(word):
    if word in names.words('male.txt'):
        return 1
    elif word in names.words('female.txt'):
        return 1
    elif GeoText(word):
        return 1
    else:
        return 0

def prec_word(words):
    lex_cat = set()
    for line in words:
        if len(line.split())>1:
            lex_cat.add(line.split()[1])
    #to get a order
    lex_cat = list(lex_cat)

    prec_cat = ""
    for line in words:
        if len(line.split())>1:
            splitted = line.split()
            if splitted[1] == prec_cat:
                prec_cat = splitted[1]
                return 1
            else:
                prec_cat = splitted[1]
                return 0
def get_lex(words):
    lex = set()
    for line in words:
        if len(line.split())>1:
            splitted = line.split()
            lex.add(splitted[0])
    return list(lex)
lexical = get_lex(open("Dutch/ned.train").readlines())
def features_lexical(words):
    len_lex = len(lexical)
    score = [0]*len_lex

    if words.split()[0] in lexical:
        ind = lexical.index(words.split()[0])
        score[ind] = 1
    return score

def get_classes(words):
    classes = set()
    for line in words:
        if len(line.split())>1:
            splitted = line.split()
            classes.add(splitted[0])
    return list(classes)
classes = get_classes(open("Dutch/ned.train").readlines())
prevClas = []
def feature_class(words):
    len_classes = len(classes)
    score = [0] * len_classes

    if len(prevClas) >0:
        ind = classes.index(prevClas[len(prevClas)-1])
        score[ind] = 1
        prevClas.append(words.split()[1])
    return score


def feature_AllCap(word):
    if word.isupper():
        return 1
    else:
        return 0
def feature_containDigits(word):
    return any(i.isdigit() for i in word)
def feature_hyphened(word):
    hyphenated = re.findall(r'\w+-\w+[-\w+]*', word)
    if len(hyphenated)>1:
        return 1
    else:
        return 0
