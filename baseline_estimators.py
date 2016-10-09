
from __future__ import division
import numpy as np

class BaselineClassifier:
    def __init__(self, method, value=None):
        self.method = method
        self.value = value
        
    def fit(self, X, y):
        n = len(y)
        ys = set(y)
        counts = np.histogram(y, list(set(y) | set([max(y)+1])))[0]
        
        if self.method == 'random':
            self.ys = ys
        elif self.method == 'scaled':
            self.ys = ys
            self.probabilities = counts / n
        elif self.method == 'majority':
            self.value = list(ys)[np.argmax(counts)]
        elif self.method == 'fixed':
            pass
            
    def predict(self, X):
        n = len(X)
        if self.method == 'random':
            yh = np.random.choice(a = list(self.ys), size = n)
        elif self.method == 'scaled':
            yh = np.random.choice(a = list(self.ys), size = n, p = self.probabilities)
        elif self.method in ('majority', 'fixed'):
            yh = [self.value] * n
        return yh
        
class BaselineRegressor:
    def __init__(self, method, value=None):
        self.method = method
        self.value = value

    def fit(self, X, y):
        if self.method == 'mean':
            self.value = np.mean(y)
        elif self.method == 'median':
            self.value = np.median(y)
        elif self.method == 'random':
            self.value = (min(y), max(y))
        elif self.method == 'normal':
            self.mean = np.mean(y)
            self.sd = np.std(y)
        elif self.method == 'fixed':
            pass

    def predict(self, X):
        n = len(X)
        if self.method in ['mean','median','fixed']:
            yh = np.array([self.value] * n)
        elif self.method == 'random':
            yh = self.value[0] + (self.value[1] - self.value[0]) * np.random.random(n)
        elif self.method == 'normal':
            yh = np.random.normal(self.mean, self.sd, n)
        return yh    
            
            