import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from utils import DataManager
from utils import get_revenue_pair, get_label_pair, get_adr_pair
import joblib
import torch

# light GBM / XGBoost

class CancelModelWrapper(object):
    def __init__(self, model, config, **args):
        # if model == 'RFC':
        #     self.model = RandomForestClassifier(**config[model])
        # elif model == 'ADBC':
        #     self.model = AdaBoostClassifier(**config[model])
        # elif model == 'SVC':
        #     self.model = SVC(**config[model])
        if model == 'RFC':
            self.model = RandomForestClassifier(**config[model])
        elif model == 'ADBC':
            self.model = AdaBoostClassifier(**config[model])
        elif model == 'SVC':
            self.model = SVC(**config[model])

    def save(self, path):
        joblib.dump(self.model, path)
        
    def train(self, X_tra):
        X_tra, Y_tra, _ = get_adr_pair(X_tra)
        assert len(X_tra) == len(Y_tra)
        self.model.fit(X_tra, Y_tra)

    def predict(self, X_tst):
        return self.model.predict(X_tst)
        
    def load(self, path):
        self.model = joblib.load(path)


class ModelWrapper(object):
    def __init__(self, model, config, **args):
        self.IsStructLearning = True if model not in [
            'RF', 'ADB', 'SVR'] else False
        if model == 'RF':
            self.model = RandomForestRegressor(**config[model])
        elif model == 'ADB':
            self.model = AdaBoostRegressor(**config[model])
        elif model == 'SVR':
            self.model = SVR(**config[model])

    def save(self, path):
        if not self.IsStructLearning:
            joblib.dump(self.model, path)
        else:
            torch.save(self.model, path)

    def train(self, X_tra):
        if not self.IsStructLearning:
            X_tra, Y_tra = get_revenue_pair(X_tra)
            assert len(X_tra) == len(Y_tra)
            self.model.fit(X_tra, Y_tra)

    def predict(self, X_tst, output='label'):
        assert output in ['label', 'revenue']
        if not self.IsStructLearning:
            if output == 'label':
                Y_pre = []
                for x in X_tst:
                    Y_pre.append(self.model.predict(x[1]))
                return [y.sum() // 10000 for y in Y_pre]
            else:
                return self.model.predict(X_tst)
        else:
            Y_pre = []
            for x in X_tst:
                Y_pre.append(self.model.predict(x[1]))
            return [y.sum() // 10000 for y in Y_pre] if output == 'label' else Y_pre

    def load(self, path):
        if not self.IsStructLearning:
            self.model = joblib.load(path)
