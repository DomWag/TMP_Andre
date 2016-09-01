import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn as sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

def get_accuracy(y_ref, y_pred):
    return accuracy_score(y_ref, y_pred)
def get_F1(y_ref, y_pred):
    return sklearn.metrics.f1_score(y_ref, y_pred)
def get_confusionMatrix(y_ref, y_pred):
    return confusion_matrix(y_ref, y_pred)


def main_start():
    print("There are three files per language: dutch, spain")
    print("_train, _testa, _testb")

    exits = False
    global lri
    while(not exits):
        s = raw_input("What do you want to do?")
        ss = s.split(" ")
        if "train" in ss[1]:
            if len(ss)==3:
                df = pd.read_csv(ss[2], header=None)
                range_end = len(next(csv.reader(open(ss[2]), delimiter=',')))
            else:
                df = pd.read_csv(ss[3], header=None)
                range_end = len(next(csv.reader(open(ss[3]), delimiter=',')))


            depp = []

            for x in range(0, range_end-1):
                depp.append(x)
            # print(depp)
            X = df.iloc[:, depp].values
            y = df.iloc[:, range_end-1].values

            y = np.where(y == 'O', 0, 1)
            lri = LogisticRegression(n_iter=500, eta=0.2).fit(X, y)
            if "learning_curve" in s:
                plt.plot(range(1, len(lri.cost_) + 1), np.log10(lri.cost_))
                plt.xlabel('Epochs')
                plt.ylabel('Cost')
                plt.title('Logistic Regression - Learning rate 0.01')

                plt.tight_layout()
                plt.show()
        elif "test" in ss[1]:
            if lri is not None:
                df = pd.read_csv(ss[2], header=None)
                range_end = len(next(csv.reader(open(ss[2]), delimiter=',')))
                depp = []

                for x in range(0, range_end - 1):
                    depp.append(x)
                X = df.iloc[:, depp].values
                y_ref = df.iloc[:, range_end-1].values

                y_ref = np.where(y_ref == 'O', 0, 1)
                y_pred = lri.predict(X)

                print (get_accuracy(y_ref, y_pred))
                print (get_F1(y_ref, y_pred))
                print (get_confusionMatrix(y_ref, y_pred))
        else:
            exits = True
main_start()