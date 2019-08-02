import numpy as np
from functools import partial
from sklearn import metrics
import scipy as sp

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(-loss_partial(self.coef_['x']))

    def predict(self, X, coef=None):
        if coef == None:
            coef = [0.5, 1.5, 2.5, 3.5]
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

def kappa_loss(output, y):
    y_hat = prediction(output)
    kappa = metrics.cohen_kappa_score(y, y_hat, weights='quadratic')
    return kappa

def prediction(output):
    y_hat = np.copy(output)
    coef = [0.5, 1.5, 2.5, 3.5]
    for i, pred in enumerate(y_hat):
        if pred < coef[0]:
            y_hat[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            y_hat[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            y_hat[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            y_hat[i] = 3
        else:
            y_hat[i] = 4
    return y_hat

# @[optimizer for quadratic weighted kappa](https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa)
