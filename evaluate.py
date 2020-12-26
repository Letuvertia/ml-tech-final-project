import numpy as np
from utils import get_label_pair, get_revenue_pair, get_adr_pair
from sklearn.metrics import mean_absolute_error

class Grader(object):
    def __init__(self, X_tst):
        self.data_label, self.label = get_label_pair(X_tst)
        self.data_revenue, self.revenue = get_revenue_pair(X_tst)
        assert len(self.data_label) == len(self.label) and len(self.data_revenue) == len(self.revenue)
        
    def eval_revenue(self, model):
        revenue_pre = model.predict(self.data_revenue, output='revenue')
        return mean_absolute_error(revenue_pre, self.revenue)
        
    def eval_cancel_rate(self, model):
        cancel_pre = model.predict(self.data_revenue, output='revenue')[self.revenue == 0] > 10000
        return cancel_pre.mean()

    def eval_mae(self, model):
        label_pre = model.predict(self.data_label, output='label')
        return mean_absolute_error(label_pre, self.label)