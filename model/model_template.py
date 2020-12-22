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
