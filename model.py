import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from utils import DataManeger, get_revenue_pair, get_label_pair, get_label_pair, get_adr_pair
import joblib
import torch

# light GBM / XGBoost

class CancelModel(object):
    def __init__(self, model, config, **args):
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
    def __init__(self, args, config):
        self.IsStructLearning = True if args.reg_model not in [
            'RFR', 'ADBR', 'SVR', 'DNN'] else False
        if args.reg_model == 'RFR':
            self.model = RandomForestRegressor(**config[args.reg_model])
        elif args.reg_model == 'ADBR':
            self.model = AdaBoostRegressor(**config[args.reg_model])
        elif args.reg_model == 'SVR':
            self.model = SVR(**config[args.reg_model])
        
        self.train_task = args.train_task
        assert self.train_task in ['adr', 'revenue', 'label']
        self.cancel_model = None
        if self.train_task == 'adr' and args.can_ckpt is not None:
            self.cancel_model = joblib.load(args.cancel_model)
            
    def save(self, path):
        if not self.IsStructLearning:
            joblib.dump(self.__dict__, path)
        else:
            # TO BE DONE
            pass
            # torch.save(self.model, path)

    def train(self, X_tra):
        if not self.IsStructLearning:
            if self.train_task == 'revenue':
                X_tra, Y_tra = get_revenue_pair(X_tra)
            elif self.train_task == 'adr':
                X_tra, _, Y_tra = get_adr_pair(X_tra)

            assert len(X_tra) == len(Y_tra)
            self.model.fit(X_tra, Y_tra)

    def predict(self, X_tst, output='label'):
        assert output in ['label', 'cancel', 'revenue']
        if not self.IsStructLearning:
            if self.train_task == 'adr':
                def get_revenue(X_tst):
                    if self.cancel_model is not None:
                        stays_in_weekend_nights = X_tst[:, -2]
                        stays_in_week_nights = X_tst[:, -1]
                        cancel_pre = 1 - self.cancel_model.predict(X_tst)
                        return self.model.predict(X_tst) * (stays_in_weekend_nights + stays_in_week_nights) * cancel_pre
                    else:
                        print("cancel model haven't been loaded!")
                        return None
            if output == 'label':
                # preparing data
                Y_pre = []
                place = []
                counter = 0
                for i in range(len(X_tst)):
                    place.append(np.arange(len(X_tst[i][1])).astype(int) + counter)
                    counter += len(place[-1])
                X_tst_cat = np.vstack([x[1] for x in X_tst])
                           
                if self.train_task == 'revenue':
                    if self.cancel_model is not None:
                        Y_tst_cat = self.model.predict(X_tst_cat) * (1 - self.cancel_model.predict(X_tst_cat))
                        for p in place:
                            Y_pre.append(Y_tst_cat[p])
                    else:
                        Y_tst_cat = self.model.predict(X_tst_cat)
                        for p in place:
                            Y_pre.append(Y_tst_cat[p])

                elif self.train_task == 'adr':
                    Y_tst_cat = get_revenue(X_tst_cat)
                    for p in place:
                            Y_pre.append(Y_tst_cat[p])
                return [y.sum() // 10000 for y in Y_pre]
            elif output == 'revenue':
                if self.train_task == 'revenue':
                    if self.cancel_model is not None:
                        return self.model.predict(X_tst) * (1 - self.cancel_model.predict(X_tst))
                    else:
                        return self.model.predict(X_tst)
                elif self.train_task == 'adr':
                    return get_revenue(X_tst)
            else:
                assert self.cancel_model is not None
                return self.cancel_model.predict(X_tst)
        else:
            pass
            # TO BE DONE
            # Y_pre = []
            # for x in X_tst:
            #     Y_pre.append(self.model.predict(x[1]))
            # return [y.sum() // 10000 for y in Y_pre] if output == 'label' else Y_pre

    def load(self, path):
        if not self.IsStructLearning:
            params = joblib.load(path)
            for att in params:
                setattr(self, att, params[att])
            
