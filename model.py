import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from utils import get_label_pair, get_revenue_pair
import joblib
import torch

# light GBM / XGBoost / Gradient boost

class ModelTemplate(object):
    """
    Below are the functions that will be called in train.py and test.py
    """
    def __init__(self, **args):
        pass
    
    def save_model(self, save_model_path=''):
        pass

    def train_model(self, train_data_loader, val_data_loader):
        pass

    def predict_model(self, test_data):
        pass

    def eval_model(self, test_data_loader):
        pass

    def load_initial_weight(self, model_path):
        pass


class ModelWrapper(object):
    def __init__(self, model, config, **args):
        self.IsStructLearning = True if model not in ['RF', 'ADB', 'SVR'] else False

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

    def load(self, path):
        if not self.IsStructLearning:
            self.model = joblib.load(path)
