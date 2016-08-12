import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn as sklearn


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
        logit = -y.dot(np.log(y_val + 1e-7)) - ((1 - y).dot(np.log(1 - y_val + 1e-7)))
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

df = pd.read_csv("scoring_train_small.csv", header=None)

#y = df.iloc[:, 9].values
y = df.iloc[:, 5].values

y = np.where(y == 'O', 0, 1)
counter = 0
for i in y:
    if i == 1:
        counter +=1
print(counter)
#X = df.iloc[:, [0,1,2,3,4,5,6,7,8]].values
depp = []
for x in range(0,5):
    depp.append(x)
print(depp)
X = df.iloc[:, depp].values

lr = LogisticRegression(n_iter=500, eta=0.2).fit(X, y)
plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
print(lr.cost_)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Logistic Regression - Learning rate 0.01')

plt.tight_layout()
plt.show()

df2 = pd.read_csv("/home/dominik/projects_save/scoring_testa2_ex_small.csv", header=None)
X2 = df2.iloc[:, [0,1,2,3,4]].values
y_ref = df2.iloc[:, 5].values

y_ref = np.where(y_ref == 'O', 0, 1)
y_pred = lr.predict(X2)
counter = 0
for i in y_pred:
    if i == 1:
        counter +=1
print (y_pred)
print(counter)
def get_accuracy():
    #devide the number of right guesses by the length of the list of right guesses

    if (len([i for i, j in zip(y_ref, y_pred) if i == j]))>0:
        return (len([i for i, j in zip(y_ref, y_pred) if i == j])) / len(y_ref)
    else:
        return 0
y_ref = [0,1,2,3,4,5]
y_pred =[0,1,2,3,5,4]
def get_F1():
    return sklearn.metrics.f1_score(y_ref, y_pred)
print(get_F1())