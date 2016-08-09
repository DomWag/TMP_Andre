import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

df = pd.read_csv('scoring.csv', header=None)

#y = df.iloc[:, 9].values
y = df.iloc[:, 5].values
#for idx, a in enumerate(y):
    # if a == 'O':
    #     y[idx] = 0
    # elif a=='B-PER':
    #     y[idx] = 1
    # elif a=='I-PER':
    #     y[idx] = 2
    # elif a=='B-LOC':
    #     y[idx] = 3
    # elif a=='I-LOC':
    #     y[idx] = 4
    # elif a=='B-ORG':
    #     y[idx] = 5
    # elif a=='I-ORG':
    #     y[idx] = 6
    # elif a=='B-MISC':
    #     y[idx] = 7
    # elif a=='I-MISC':
    #     y[idx] = 8
    # else:
    #     y[idx] = 0
# for idx, a in enumerate(y):
#     if a == 'O':
#         y[idx] = 0
#     else:
#         y[idx] = 1
y = np.where(y == 'O', 0, 1)
#X = df.iloc[:, [0,1,2,3,4,5,6,7,8]].values
X = df.iloc[:, [0,1,2,3,4]].values

lr = LogisticRegression(n_iter=500, eta=0.2).fit(X, y)
plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Logistic Regression - Learning rate 0.01')

plt.tight_layout()
plt.show()
