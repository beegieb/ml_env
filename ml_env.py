
import sys
from copy import copy
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, LSHForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, \
        roc_auc_score, mean_absolute_error, r2_score, explained_variance_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, \
        Perceptron, PassiveAggressiveClassifier, LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from baseline_estimators import BaselineClassifier, BaselineRegressor

def getts():
    return dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    
classifiers = dict(
        lr = LogisticRegression()
        ,lrcv = LogisticRegressionCV()
        ,sgd = SGDClassifier()
        ,perceptron = Perceptron()
        ,pac = PassiveAggressiveClassifier()
        ,lsvc = LinearSVC(C=5)
        ,dt = DecisionTreeClassifier()
        ,rf = RandomForestClassifier(max_depth=50, n_estimators=100, min_samples_split=20)
        ,gbm = GradientBoostingClassifier(max_depth=20, n_estimators=50, min_samples_split=20)
        ,gs_lr = GridSearchCV(
            estimator = LogisticRegression()
            ,param_grid = dict(
                C = [1, 3, 5, 10]
                ,penalty = ['l1','l2']
                )
            ,cv = 3
            )
        ,gs_dt = GridSearchCV(
            estimator = DecisionTreeClassifier()
            ,param_grid = dict(
                max_depth = [5, 7, 10]
                ,min_samples_split = [10, 20, 30]
                )
            ,cv = 3
            )
        ,p__pca_lr = Pipeline([('pca', PCA(n_components=1200)), ('lr', LogisticRegression(C=1, dual=False))])
        )


classifiers['vote_hard'] = VotingClassifier(
    estimators=[
        ('lr', classifiers['lr'])
        ,('rf', classifiers['rf'])
        ,('sgd', classifiers['sgd'])
        ]
    ,voting='hard'
    )

baseline_classifiers = dict(
        b_random = BaselineClassifier(method='random')
        ,b_scaled = BaselineClassifier(method='scaled')
        ,b_majority = BaselineClassifier(method='majority')
        ,b_true = BaselineClassifier(method='fixed', value=1)
        ,b_false = BaselineClassifier(method='fixed', value=0)
        )    
        
regressors = dict(
    linr = LinearRegression()
    ,ridge = Ridge()
    ,lasso = Lasso()
    ,gbr = GradientBoostingRegressor()
    ,gs_ridge = RidgeCV(alphas = [1e-2,1e-1,1,1e+1,1e+2,1e+3], cv=3)
    ,gs_lasso = LassoCV(alphas = [1e-2,1e-1,1,1e+1,1e+2,1e+3], cv=3)
    )
        
baseline_regressors = dict(
        b_mean = BaselineRegressor('mean')
        ,b_median = BaselineRegressor('median')
        ,b_random = BaselineRegressor('random')
        ,b_normal = BaselineRegressor('normal')
        ,b_zero = BaselineRegressor('fixed', 0)
        )
        

class MLEnv():
    def __init__(self, type):
        self.type = type
        self.models = dict()
        self.performance = dict()
        
    def add_model(self, model_name, estimator):
        self.models[model_name] = copy(estimator)
        
    def del_model(self, model_name):
        del self.models[model_name]
        if model_name in self.performance:
            del self.performance[model_name]

    def fit_models(self, X, y):
        for model in self.models.keys():
            print getts() + ' - INFO - Fitting: ' + model
            self.models[model].fit(X, y)
        print getts() + ' - INFO - DONE'

    def eval_models(self, X, y, debug = False):
        for model in self.models.keys():
            if debug:
                print getts() + ' - INFO - Applying: ' + model
            yh = self.models[model].predict(X)
            if self.type == 'regression':
                yh = yh + np.float64(sys.float_info.epsilon)
            
            if debug:
                print getts() + ' - INFO - Evaluating: ' + model
            if self.type in ('binary','multiclass'):
                self.performance[model] = dict(
                    accuracy = accuracy_score(y, yh)
                    ,precision = precision_score(y, yh)
                    ,recall = recall_score(y, yh)
                    )
            elif self.type in ('regression'):
                self.performance[model] = dict(
                    mae = mean_absolute_error(y, yh)
                    ,r2 = r2_score(y, yh)
                    ,explained_variance = explained_variance_score(y, yh)
                    ,rmse = np.sqrt(mean_squared_error(y, yh))
                    )
            if self.type == 'binary':
                self.performance['mcorr'] = matthews_corrcoef(y, yh)
                self.performance['roc_auc'] = roc_auc_score(y, yh)
        print getts() + ' - INFO - DONE'
        
    def compare_models(self, a, b):
        x = [i[a] for i in self.performance.values()]
        y = [i[b] for i in self.performance.values()]
        plt.scatter(x, y)
        plt.xlabel(a)
        plt.ylabel(b)
        #plt.xlim(0,1)
        #plt.ylim(0,1)
        plt.title('Model Comparison')
        xys = zip(x,y)
        for ind, txt in enumerate(self.performance.keys()):
            plt.annotate(s = txt, xy = xys[ind])
        
        plt.show()
        
