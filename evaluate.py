import numpy as np
from utils import get_label_pair, get_revenue_pair, get_adr_pair
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import classification_report

class Grader:
    def __init__(self, X_tst):
        self.data_label, self.label = get_label_pair(X_tst)
        self.data_revenue, self.revenue = get_revenue_pair(X_tst)
        _, self.cancel, self.adr = get_adr_pair(X_tst)
        self.revenue = self.revenue * (1 - self.cancel)
        assert len(self.data_label) == len(self.label) and len(self.data_revenue) == len(self.revenue)
        
    def eval_revenue(self, model):
        revenue_pre = model.predict(self.data_revenue, output='revenue')
        return mean_absolute_error(revenue_pre, self.revenue)
        
    def eval_cancel_error_rate(self, model, IsCancelModel=False):
        """
        Return: 3 floats
        1. accuracy
        2. the precentage of how many cases of "0->1" in all 0's (i.e. one less revenue)
        3. the precentage of how many cases of "1->0" in all 1's (i.e. one more revenue)
        """
        if IsCancelModel:
            cancel_pre = model.predict(self.data_revenue)
            #print(cancel_pre.shape)
            report = classification_report(self.cancel, cancel_pre, output_dict=True)
            #print(report)
            return 1-report['accuracy'], 1-report['0.0']['recall'], 1-report['1.0']['recall']
            #return 1 - accuracy_score(cancel_pre, self.cancel)
        else:
            assert model.train_task != 'label' 
            if model.cancel_model is not None:
                cancel_pre = model.predict(self.data_revenue, output='cancel')
                return 1 - accuracy_score(cancel_pre, self.cancel)
            else:
                print("cancel model haven't been loaded!")
                return None
            
    def eval_mae(self, model):
        label_pre = model.predict(self.data_label, output='label')
        print(self.label)
        return mean_absolute_error(label_pre, self.label)
