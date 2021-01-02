import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# from utils import DataManager, get_revenue_pair, get_label_pair, get_label_pair, get_adr_pair
from utils import DataManager, get_revenue_pair, get_label_pair, get_label_pair, get_adr_pair, RegressorDataset
import joblib
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math

# light GBM / XGBoost
from sklearn.metrics import mean_absolute_error

def Swish(x):
    return x * torch.sigmoid(x)

class LinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, activation='LeakyReLU', dropout=None):
        super(LinearLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.dropout = dropout

        if activation == 'Swish':
            self.act = Swish
        else:
            self.act = eval(f'torch.nn.{activation}()')

    def forward(self, features):
        predicted = self.linear(features)
        predicted = self.act(predicted)

        if self.dropout is not None:
            predicted = self.dropout(predicted)
        
        return predicted

class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = torch.nn.Linear(in_channels, key_size)
        self.linear_keys = torch.nn.Linear(in_channels, key_size)
        self.linear_values = torch.nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input, mask):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        
        #import pdb; pdb.set_trace()
        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        values = self.linear_values(input) # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp = F.softmax(temp / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values) # shape: (N, T, value_size)
        return temp * mask

class DNNRegressor:
    def __init__(self, input_dim=872, onehot_dim=None, hidden_layer=[256], use_emb=False, opt='Adam', activation='LeakyReLU',
                 dropout_list=None, batch_size=64, total_step=1000, lr=1e-3, n_jobs=6, device="cuda:0", lead_time_idx=None):
        self.use_emb = use_emb
        if self.use_emb:
            assert onehot_dim is not None
            self.onehot_dim = onehot_dim
            self.emb = torch.nn.Embedding(
                onehot_dim + 1, hidden_layer[0], padding_idx=0)
            first_layer = LinearLayer(input_dim - onehot_dim, hidden_layer[0])
        else:
            first_layer = LinearLayer(input_dim, hidden_layer[0])

        last_layer = torch.nn.Linear(hidden_layer[-1], 1)
        if dropout_list is not None:
            self.model = torch.nn.ModuleList(
                [first_layer]+[LinearLayer(hidden_layer[i-1], hidden_layer[i], activation, torch.nn.Dropout(dropout_list[i-1])) for i in range(1, len(hidden_layer))]+[last_layer])
        else:
            self.model = torch.nn.ModuleList(
                [first_layer]+[LinearLayer(hidden_layer[i-1], hidden_layer[i], activation) for i in range(1, len(hidden_layer))]+[last_layer])
        self.batch_size = batch_size
        self.total_step = total_step
        self.lr = lr
        self.n_jobs = n_jobs
        self.opt = opt
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        self.lead_time_idx = lead_time_idx

    def forward(self, model, x):
        for i in range(len(model)):
            m = model[i]
            if i == 0 and self.use_emb:
                x_num = x[:, self.onehot_dim:]
                x_cat = (x[:, :self.onehot_dim] > 0) * torch.cat([torch.range(
                    1, self.onehot_dim).unsqueeze(0)] * x.shape[0], dim=0).long().to(self.device)
                x = m(x_num) + self.emb(x_cat).mean(dim=1)
            else:
                x = m(x)
        
        return x.squeeze(-1)

    def get_hidden(self, x):
        for i in range(len(self.model) - 1):
            m = self.model[i]
            if i == 0 and self.use_emb:
                x_num = x[:, self.onehot_dim:]
                x_cat = (x[:, :self.onehot_dim] > 0) * torch.cat([torch.range(
                    1, self.onehot_dim).unsqueeze(0)] * x.shape[0], dim=0).long().to(self.device)
                x = m(x_num) + self.emb(x_cat).mean(dim=1)
            else:
                x = m(x)
        
        return x.squeeze(-1)

    def feature_extract(self, X_tst):
        self.model = self.model.to(self.device)
        testloader = DataLoader(RegressorDataset(X_tst),
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.n_jobs)
        with torch.no_grad():
            Y_pre = []
            for x in testloader:
                Y_pre.append(self.get_hidden(x.to(
                    self.device)).detach().cpu().numpy())
        return np.vstack(Y_pre)

    def fit(self, X_tra, Y_tra, grader=None):
        trainloader = DataLoader(RegressorDataset(X_tra, Y_tra),
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.n_jobs)

        global_steps = 0
        optimizer = eval(
            f'torch.optim.{self.opt}(self.model.parameters(), lr=self.lr)')
        criterion = torch.nn.L1Loss()
        if self.use_emb:
            self.emb = self.emb.to(self.device)
        self.model = self.model.to(self.device)

        if grader is not None:
            (X_val, _), Y_val = grader.data_revenue, grader.adr
            loss_record = []
        while True:
            for (x, y) in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pre = self.forward(self.model, x)
                loss = criterion(y_pre, y)
                loss.backward()
                if grader is not None:
                    loss_record.append(loss.item())
                    if grader is not None and global_steps % 1000 == 0 and global_steps != 0:
                        print('steps: {:d} | tra_adr: {:.4f} | val_adr: {:.4f}'.format(
                            global_steps, np.mean(loss_record), mean_absolute_error(self.predict(X_val), Y_val)))
                        loss_record = []

                # print(global_steps, loss.item())
                
                optimizer.step()
                optimizer.zero_grad()
                global_steps += 1
                if self.total_step <= global_steps:
                    self.model = self.model.cpu()
                    self.model.eval()
                    return

    def predict(self, X_tst):
        self.model = self.model.to(self.device)
        self.model.eval()

        testloader = DataLoader(RegressorDataset(X_tst),
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.n_jobs)
        with torch.no_grad():
            Y_pre = []
            for x in testloader:
                Y_pre.append(self.forward(self.model, x.to(
                    self.device)).detach().cpu().numpy())
        self.model.train()
        return np.hstack(Y_pre)

class OrdinalClassifier:
    def __init__(self, architecture, opt='Adam', activation='LeakyReLU',
                 dropout_list=None, batch_size=64, total_step=1000, lr=1e-3, n_jobs=6, device="cuda:0"):
        self.use_emb = use_emb
        if self.use_emb:
            assert onehot_dim is not None
            self.onehot_dim = onehot_dim
            self.emb = torch.nn.Embedding(
                onehot_dim + 1, hidden_layer[0], padding_idx=0)
            first_layer = LinearLayer(input_dim - onehot_dim, hidden_layer[0])
        else:
            first_layer = LinearLayer(input_dim, hidden_layer[0])

        last_layer = torch.nn.Linear(hidden_layer[-1], 1)
        if dropout_list is not None:
            self.model = torch.nn.ModuleList(
                [first_layer]+[LinearLayer(hidden_layer[i-1], hidden_layer[i], activation, torch.nn.Dropout(dropout_list[i-1])) for i in range(1, len(hidden_layer))]+[last_layer])
        else:
            self.model = torch.nn.ModuleList(
                [first_layer]+[LinearLayer(hidden_layer[i-1], hidden_layer[i], activation) for i in range(1, len(hidden_layer))]+[last_layer])
        self.batch_size = batch_size
        self.total_step = total_step
        self.lr = lr
        self.n_jobs = n_jobs
        self.opt = opt
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        self.lead_time_idx = lead_time_idx

    def forward(self, model, x):
        for i in range(len(model)):
            m = model[i]
            if i == 0 and self.use_emb:
                x_num = x[:, self.onehot_dim:]
                x_cat = (x[:, :self.onehot_dim] > 0) * torch.cat([torch.range(
                    1, self.onehot_dim).unsqueeze(0)] * x.shape[0], dim=0).long().to(self.device)
                x = m(x_num) + self.emb(x_cat).mean(dim=1)
            else:
                x = m(x)
        
        return x.squeeze(-1)

    def get_hidden(self, x):
        for i in range(len(self.model) - 1):
            m = self.model[i]
            if i == 0 and self.use_emb:
                x_num = x[:, self.onehot_dim:]
                x_cat = (x[:, :self.onehot_dim] > 0) * torch.cat([torch.range(
                    1, self.onehot_dim).unsqueeze(0)] * x.shape[0], dim=0).long().to(self.device)
                x = m(x_num) + self.emb(x_cat).mean(dim=1)
            else:
                x = m(x)
        
        return x.squeeze(-1)

    def feature_extract(self, X_tst):
        self.model = self.model.to(self.device)
        testloader = DataLoader(RegressorDataset(X_tst),
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.n_jobs)
        with torch.no_grad():
            Y_pre = []
            for x in testloader:
                Y_pre.append(self.get_hidden(x.to(
                    self.device)).detach().cpu().numpy())
        return np.vstack(Y_pre)

    def fit(self, X_tra, Y_tra, grader=None):
        trainloader = DataLoader(RegressorDataset(X_tra, Y_tra, self.lead_time_idx),
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.n_jobs)

        global_steps = 0
        optimizer = eval(
            f'torch.optim.{self.opt}(self.model.parameters(), lr=self.lr)')
        criterion = torch.nn.L1Loss()
        if self.use_emb:
            self.emb = self.emb.to(self.device)
        self.model = self.model.to(self.device)

        if grader is not None:
            (X_val, _), Y_val = grader.data_revenue, grader.adr
            loss_record = []
        while True:
            for (x, y) in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pre = self.forward(self.model, x)
                loss = criterion(y_pre, y)
                loss.backward()
                if grader is not None:
                    loss_record.append(loss.item())
                    if grader is not None and global_steps % 1000 == 0 and global_steps != 0:
                        print('steps: {:d} | tra_adr: {:.4f} | val_adr: {:.4f}'.format(
                            global_steps, np.mean(loss_record), mean_absolute_error(self.predict(X_val), Y_val)))
                        loss_record = []

                # print(global_steps, loss.item())
                
                optimizer.step()
                optimizer.zero_grad()
                global_steps += 1
                if self.total_step <= global_steps:
                    self.model = self.model.cpu()
                    self.model.eval()
                    return

    def predict(self, X_tst):
        self.model = self.model.to(self.device)
        self.model.eval()

        testloader = DataLoader(RegressorDataset(X_tst),
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.n_jobs)
        with torch.no_grad():
            Y_pre = []
            for x in testloader:
                Y_pre.append(self.forward(self.model, x.to(
                    self.device)).detach().cpu().numpy())
        self.model.train()
        return np.hstack(Y_pre)

class CancelModel(object):
    def __init__(self, model, config, filter_all, use_onehot, **args):
        if model == 'RFC':
            self.model = RandomForestClassifier(**config[model])
        elif model == 'ADBC':
            self.model = AdaBoostClassifier(**config[model])
        elif model == 'SVC':
            self.model = SVC(**config[model])
        self.drop_list = config['base']['cancel_drop_list']
        self.filter_all = filter_all
        self.use_onehot = use_onehot

    def save(self, path):
        joblib.dump(self.__dict__, path)

    def train(self, X_tra):
        X_tra, Y_tra, _ = get_adr_pair(X_tra)
        assert len(X_tra) == len(Y_tra)
        self.model.fit(X_tra, Y_tra)

    def predict(self, X_tst):
        return self.model.predict(X_tst)

    def load(self, path):
        params = joblib.load(path)
        for att in params:
            setattr(self, att, params[att])


class ModelWrapper(object):
    def __init__(self, args, config, use_onehot, filter_all=True, input_dim=None, onehot_dim=None, lead_time_idx=None):

        self.IsStructLearning = True if args.reg_model not in [
            'RFR', 'ADBR', 'SVR', 'DNR'] else False
        if args.reg_model == 'RFR':
            self.model = RandomForestRegressor(**config[args.reg_model])
        elif args.reg_model == 'ADBR':
            self.model = AdaBoostRegressor(**config[args.reg_model])
        elif args.reg_model == 'SVR':
            self.model = SVR(**config[args.reg_model])
        elif args.reg_model == 'DNR':
            assert input_dim is not None and onehot_dim is not None
            config[args.reg_model]['input_dim'] = input_dim
            config[args.reg_model]['onehot_dim'] = onehot_dim
            if lead_time_idx is not None:
                config[args.reg_model]['lead_time_idx'] = lead_time_idx
            self.model = DNNRegressor(**config[args.reg_model])
        elif args.reg_model == 'OC':
            assert input_dim is not None and onehot_dim is not None
            config[args.reg_model]['input_dim'] = input_dim
            config[args.reg_model]['onehot_dim'] = onehot_dim
            self.model = OrdinalClassifier(**config[args.reg_model])
            
        self.filter_all = filter_all
        self.use_onehot = use_onehot

        self.drop_list = config['base']['target_drop_list']
        self.train_task = args.train_task
        assert self.train_task in ['adr', 'revenue', 'label', 'pretrain']
        self.cancel_model = None
        if self.train_task == 'adr' and args.can_ckpt is not None:
            self.cancel_model = CancelModel('RFC', config, False, False)
            self.cancel_model.load(args.can_ckpt)

    def save(self, path):
        if not self.IsStructLearning:
            joblib.dump(self.__dict__, path)

    def train(self, X_tra, grader=None):
        if not self.IsStructLearning:
            if self.train_task == 'revenue':
                X_tra, Y_tra = get_revenue_pair(X_tra)
            elif self.train_task == 'adr':
                X_tra, _, Y_tra = get_adr_pair(X_tra)
            elif self.train_task == 'pretrain':
                X_tra, Y_tra = get_lead_time_pair(X_tra)

            assert len(X_tra) == len(Y_tra)
            if grader is not None:
                self.model.fit(X_tra, Y_tra, grader)
            else:
                self.model.fit(X_tra, Y_tra)

    def predict(self, X_tst, output='label'):
        assert output in ['label', 'cancel', 'revenue', 'hidden']
        if not self.IsStructLearning:

            if self.train_task == 'adr':
                def get_revenue(X_tst):
                    if self.cancel_model is not None:
                        assert isinstance(X_tst, tuple) and len(X_tst) == 2
                        X_tst, X_tst_can = X_tst
                        cancel_pre = 1 - self.cancel_model.predict(X_tst_can)

                    stays_in_weekend_nights = X_tst[:, -2]
                    stays_in_week_nights = X_tst[:, -1]

                    if self.cancel_model is not None:
                        return self.model.predict(X_tst) * (stays_in_weekend_nights + stays_in_week_nights) * cancel_pre
                    else:
                        return self.model.predict(X_tst) * (stays_in_weekend_nights + stays_in_week_nights)

            if output == 'label':
                # preparing data
                if self.cancel_model is not None:
                    assert isinstance(X_tst, tuple) and len(X_tst) == 2
                    X_tst, X_tst_can = X_tst

                place = []
                counter = 0
                for i in range(len(X_tst)):
                    place.append(
                        np.arange(len(X_tst[i][1])).astype(int) + counter)
                    counter += len(place[-1])

                if self.cancel_model is not None:
                    X_tst_can_cat = np.vstack([x[1] for x in X_tst_can])
                X_tst_cat = np.vstack([x[1] for x in X_tst])

                if self.train_task == 'revenue':
                    if self.cancel_model is not None:
                        Y_tst_cat = self.model.predict(
                            X_tst_cat) * (1 - self.cancel_model.predict(X_tst_can_cat))
                    else:
                        Y_tst_cat = self.model.predict(X_tst_cat)

                elif self.train_task == 'adr':
                    if self.cancel_model is not None:
                        X_tst_cat = (X_tst_cat, X_tst_can_cat)
                    Y_tst_cat = get_revenue(X_tst_cat)

                Y_pre = [Y_tst_cat[p] for p in place]
                return [min(9.0, max(0.0, y.sum() // 10000)) for y in Y_pre]

            elif output == 'revenue':
                if self.train_task == 'revenue':
                    if self.cancel_model is not None:
                        assert isinstance(X_tst, tuple) and len(X_tst) == 2
                        X_tst, X_tst_can = X_tst
                        return self.model.predict(X_tst) * (1 - self.cancel_model.predict(X_tst_can))
                    else:
                        return self.model.predict(X_tst)

                elif self.train_task == 'adr':
                    return get_revenue(X_tst)
            elif output == 'cancel':
                _, X_tst = X_tst
                assert self.cancel_model is not None
                return self.cancel_model.predict(X_tst)

            elif output == 'hidden':
                place = []
                counter = 0
                for i in range(len(X_tst)):
                    place.append(
                        np.arange(len(X_tst[i][1])).astype(int) + counter)
                    counter += len(place[-1])

                X_tst_cat = np.vstack([x[1][:, :-3] for x in X_tst])
                X_hd_cat = self.model.feature_extract(X_tst_cat)
                X_hd = [X_hd_cat[p] for p in place]
                return [[X_tst[i][0], np.hstack([X_hd[i], X_tst[i][1][:, -3:]])] for i in range(len(X_tst))]
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
