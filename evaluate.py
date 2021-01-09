import numpy as np
from utils import get_label_pair, get_revenue_pair, get_adr_pair, get_rev_lst_pair
from sklearn.metrics import mean_absolute_error, accuracy_score

class Grader:
    def __init__(self, X_tst):
        self.data_label, self.label = get_label_pair(X_tst)
        self.data_revenue, self.revenue = get_revenue_pair(X_tst)
        self.data_OC, self.adr_OC, self.revenue_OC  = get_rev_lst_pair(X_tst)
        _, self.cancel, self.adr = get_adr_pair(X_tst)
        self.revenue = self.revenue * (1 - self.cancel)
        # assert len(self.data_label) == len(self.label) and len(self.data_revenue) == len(self.revenue)
        
    def eval_revenue(self, model):
        if model.train_task == 'label':
            revenue_pre = model.predict(self.data_OC, output='revenue')
            revenue_pre = np.hstack(revenue_pre)
            return mean_absolute_error(revenue_pre, self.revenue)
        else:
            revenue_pre = model.predict(self.data_revenue, output='revenue')
            return mean_absolute_error(revenue_pre, self.revenue)
        
    def eval_cancel_error_rate(self, model, IsCancelModel=False):
        if IsCancelModel:
            cancel_pre = model.predict(self.data_revenue)
            return 1 - accuracy_score(cancel_pre, self.cancel)
        else:
            assert model.train_task != 'label' 
            if model.cancel_model is not None:
                cancel_pre = model.predict(self.data_revenue, output='cancel')
                return 1 - accuracy_score(cancel_pre, self.cancel)
            else:
                print("cancel model haven't been loaded!")
                return None
            
    def eval_mae(self, model):
        if model.train_task == 'label':
            label_pre = model.predict(self.data_OC, output='label')
            return mean_absolute_error(label_pre, self.label)
        else:
            label_pre = model.predict(self.data_label, output='label')
            return mean_absolute_error(label_pre, self.label)