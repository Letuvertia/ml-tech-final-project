import copy
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
# import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier, CatBoostRegressor


from utils import DataManager, get_revenue_pair, get_label_pair, get_label_pair, get_adr_pair, get_rev_lst_pair, RegressorDataset, OrdinalClassifierDataset
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
    def __init__(self, input_dim, output_dim, activation='LeakyReLU', dropout=None, bias=True):
        super(LinearLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias)
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

    def forward(self, input, mask=None):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length

        # import pdb; pdb.set_trace()
        keys = self.linear_keys(input)  # shape: (N, T, key_size)
        query = self.linear_query(input)  # shape: (N, T, key_size)
        values = self.linear_values(input)  # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(
            keys, 1, 2))  # shape: (N, T, T)
        # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = F.softmax(temp / self.sqrt_key_size, dim=1)
        temp = torch.bmm(temp, values)  # shape: (N, T, value_size)
        if mask is not None:
            return temp * mask
        return temp


class TransformerLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(TransformerLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1dBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation='LeakyReLU'):
        super(Conv1dBlock, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=kernel_size//2)
        self.act = eval(f'torch.nn.{activation}()')

    def forward(self, features):
        features = features.transpose(-1, -2).contiguous()
        predicted = self.conv(features)
        predicted = self.act(predicted)
        return predicted.transpose(-1, -2).contiguous()


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
            first_layer = LinearLayer(input_dim, hidden_layer[0], bias=False)

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
                x_cat = (x[:, :self.onehot_dim] > 0) * torch.cat([torch.arange(
                    1, self.onehot_dim + 1).unsqueeze(0)] * x.shape[0], dim=0).long().to(self.device)
                x = m(x_num) + self.emb(x_cat).mean(dim=1)
            else:
                x = m(x)

        # return x.squeeze(-1)
        return torch.clamp(x.squeeze(-1), max=370, min=-145)

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
            min_adr = 100
            best_model = copy.deepcopy(self.model)

            (X_val, _), Y_val = grader.data_revenue, grader.adr
            loss_record = []
        while True:
            for (x, y) in trainloader:
                # truncation
                y = torch.clamp(y, max=370, min=-145)

                x, y = x.to(self.device), y.to(self.device)
                y_pre = self.forward(self.model, x)
                loss = criterion(y_pre, y)
                loss.backward()
                if grader is not None:
                    loss_record.append(loss.item())
                    if grader is not None and global_steps % 1000 == 0 and global_steps != 0:
                        adr_val = mean_absolute_error(
                            self.predict(X_val), Y_val)
                        print('steps: {:d} | tra_adr: {:.4f} | val_adr: {:.4f}'.format(
                            global_steps, np.mean(loss_record), adr_val))
                        loss_record = []
                        if min_adr > adr_val:
                            min_adr = adr_val
                            best_model = copy.deepcopy(self.model)

                # print(global_steps, loss.item())

                optimizer.step()
                optimizer.zero_grad()
                global_steps += 1
                if self.total_step <= global_steps + 1:
                    if grader is not None:
                        self.model = best_model
                    self.model = self.model.cpu()
                    self.model.eval()
                    self.model = best_model
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
    def __init__(self, input_dim=872, opt='Adam', activation='LeakyReLU', use_label=False,
                 batch_size=64, total_step=1000, lr=1e-3, n_jobs=6, device="cuda:0"):

        self.extractor = torch.nn.Sequential(AttentionBlock(
            input_dim, 256, 256), AttentionBlock(256, 256, 256))

        self.use_label = use_label
        if self.use_label:
            self.label_head = torch.nn.Sequential(LinearLayer(256, 128, activation), LinearLayer(
                128, 64, activation), torch.nn.Linear(64, 1))

        self.adr_head = torch.nn.ModuleList([torch.nn.Sequential(LinearLayer(256, 64, activation), LinearLayer(
            64, 16, activation)), torch.nn.Linear(16, 1)])

        self.batch_size = batch_size
        self.total_step = total_step
        self.lr = lr
        self.n_jobs = n_jobs
        self.opt = opt
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")

    def forward(self, x, m):
        x = self.extractor(x) * m
        h = self.adr_head[0](x)
        adr = self.adr_head[1](h) * m
        adr = torch.clamp(adr, min=-145, max=370).squeeze(-1)
        if self.use_label:
            label = torch.clamp(self.label_head(
                x.sum(dim=1)), min=0.0, max=9.0).squeeze(-1)
            return (adr, label)
        return adr

    def surrogate_loss(self, y_pre, y_tar):
        if self.use_label:
            y_pre_adr, y_pre_label = y_pre
            y_tar_adr, y_tar_label = y_tar

            mask = torch.floor(y_pre_label.detach()) != y_tar
            label_loss = F.l1_loss(y_pre_label * mask, y_tar_label * mask)
            adr_loss = F.l1_loss(y_pre_adr, y_tar_adr)
            return adr_loss + label_loss
        else:
            adr_loss = F.l1_loss(y_pre, y_tar)
            return adr_loss

    def fit(self, X_tra, Y_tra, grader=None):
        dataset = OrdinalClassifierDataset(X_tra, Y_tra)
        trainloader = DataLoader(dataset,
                                 collate_fn=dataset.collate_fn,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.n_jobs)

        global_steps = 0
        self.extractor = self.extractor.to(self.device)
        self.adr_head = self.adr_head.to(self.device)
        if self.use_label:
            self.label_head = self.label_head.to(self.device)
            label_optimizer = eval(
                f'torch.optim.{self.opt}(self.label_head.parameters(), lr=self.lr)')
        optimizer = eval(
            f'torch.optim.{self.opt}(list(self.extractor.parameters()) + list(self.adr_head.parameters()), lr=self.lr)')

        if grader is not None:
            from torch.nn.utils.rnn import pad_sequence
            X_val, Y_val = grader.data_OC, grader.adr_OC
            X_val, _ = X_val
            Y_val = [torch.from_numpy(y) for y in Y_val]
            Y_val = pad_sequence(Y_val, batch_first=True)
            validset = OrdinalClassifierDataset(X_val)
            validloader = DataLoader(validset,
                                     collate_fn=validset.collate_fn,
                                     batch_size=150,
                                     shuffle=False,
                                     num_workers=self.n_jobs)
            validloader = iter(validloader)
            (X_val, M_val) = next(validloader)
            loss_record = []
        min_mae = 1.0
        while True:
            for (x, y_adr, y_label, m) in trainloader:
                x, y_adr, y_label, m = x.to(self.device), y_adr.to(
                    self.device), y_label.to(self.device), m.to(self.device)
                with torch.no_grad():
                    scalar = (y_adr.shape[0] * y_adr.shape[1] / m.sum().item())
                y_pre = self.forward(x, m)
                if self.use_label:
                    loss = self.surrogate_loss(y_pre, (y_adr, y_label)) * scalar
                else:
                    loss = self.surrogate_loss(y_pre, y_adr) * scalar
                loss.backward()
                optimizer.step()
                if self.use_label:
                    label_optimizer.step()
                    label_optimizer.zero_grad()
                optimizer.zero_grad()
                
                if grader is not None:
                    loss_record.append(loss.item())
                    if grader is not None and global_steps % 100 == 0 and global_steps != 0:
                        self.extractor.eval()
                        self.adr_head.eval()
                        with torch.no_grad():
                            if not self.use_label:
                                Y_val_pre = self.forward(X_val.to(self.device), M_val.to(
                                    self.device)).detach().cpu()
                            else:
                                Y_val_pre, Y_val_label_pre = self.forward(X_val.to(self.device), M_val.to(
                                    self.device))
                                Y_val_pre = Y_val_pre.detach().cpu()
                            adr_val = ((Y_val_pre - Y_val).abs().sum() / M_val.sum()).item()
                        print('steps: {:d} | tra_adr: {:.4f} | val_adr: {:.4f}'.format(
                            global_steps, np.mean(loss_record), adr_val))

                        self.extractor.train()
                        self.adr_head.train()
                        loss_record = []

                global_steps += 1
                if self.total_step <= global_steps:
                    self.extractor = self.extractor.cpu()
                    self.extractor.eval()
                    self.adr_head = self.adr_head.cpu()
                    self.adr_head.eval()
                    if self.use_label:
                        self.label_head = self.label_head.cpu()
                        self.label_head.eval()

                    return

    def predict(self, X_tst):
        self.extractor = self.extractor.to(self.device)
        self.adr_head = self.adr_head.to(self.device)
        self.extractor.eval()
        self.adr_head.eval()
        testset = OrdinalClassifierDataset(X_tst)
        testloader = DataLoader(testset,
                                collate_fn=testset.collate_fn,
                                batch_size=200,
                                shuffle=False,
                                num_workers=self.n_jobs)
        Y_pre = []
        with torch.no_grad():
            for (x, m) in testloader:
                x, m = x.to(self.device), m.to(self.device)
                y_pre = self.forward(x, m).detach().cpu().numpy()
                Y_pre += [y_pre[i][:m[i].sum()] for i in range(len(y_pre))]
        self.extractor.train()
        self.adr_head.train()
        return Y_pre
    
    def get_hidden(self, X_tst):
        self.extractor = self.extractor.to(self.device)
        self.adr_head = self.adr_head.to(self.device)
        self.extractor.eval()
        self.adr_head.eval()
        testset = OrdinalClassifierDataset(X_tst)
        testloader = DataLoader(testset,
                                collate_fn=testset.collate_fn,
                                batch_size=200,
                                shuffle=False,
                                num_workers=self.n_jobs)
        Y_pre = []
        with torch.no_grad():
            for (x, m) in testloader:
                x, m = x.to(self.device), m.to(self.device)
                x = self.extractor(x) * m
                y_pre = self.adr_head[0](x).detach().cpu().numpy()
                Y_pre += [y_pre[i][:m[i].sum()] for i in range(len(y_pre))]
        self.extractor.train()
        self.adr_head.train()
        return Y_pre


class CancelModel(object):
    def __init__(self, model, config, filter_all, use_onehot, **args):
        if model == 'RFC':
            self.model = RandomForestClassifier(**config[model])
        elif model == 'ADBC':
            self.model = AdaBoostClassifier(**config[model])
        elif model == 'SVC':
            self.model = SVC(**config[model])
        elif model == 'XGBC':
            self.model = XGBClassifier(**config[model])
        elif model == "CBC":
            self.model = CatBoostClassifier(**config[model])
        elif model == 'HGBC':
            self.model = HistGradientBoostingClassifier(**config[model])

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
        # return self.model.predict_proba(X_tst)[:, 1]
        return self.model.predict(X_tst)

    def load(self, path):
        params = joblib.load(path)
        for att in params:
            setattr(self, att, params[att])


class ModelWrapper(object):
    def __init__(self, args, config, use_onehot, filter_all=True, input_dim=None, onehot_dim=None, lead_time_idx=None):

        self.IsStructLearning = True if args.reg_model not in [
            'RFR', 'ADBR', 'SVR', 'DNR', 'GBR', 'CBR', 'HGBR'] else False
        if args.reg_model == 'RFR':
            self.model = RandomForestRegressor(**config[args.reg_model])
        elif args.reg_model == 'ADBR':
            self.model = AdaBoostRegressor(**config[args.reg_model])
        elif args.reg_model == 'SVR':
            self.model = SVR(**config[args.reg_model])
        elif args.reg_model == 'GBR':
            self.model = GradientBoostingRegressor(**config[args.reg_model])
        elif args.reg_model == 'HGBR':
            self.model = HistGradientBoostingRegressor(
                **config[args.reg_model])
        elif args.reg_model == 'DNR':
            assert input_dim is not None and onehot_dim is not None
            config[args.reg_model]['input_dim'] = input_dim
            config[args.reg_model]['onehot_dim'] = onehot_dim
            if lead_time_idx is not None:
                config[args.reg_model]['lead_time_idx'] = lead_time_idx
            self.model = DNNRegressor(**config[args.reg_model])
        elif args.reg_model == 'CBR':
            self.model = CatBoostRegressor(**config[args.reg_model])

        elif args.reg_model == 'OC':
            assert input_dim is not None and onehot_dim is not None
            config[args.reg_model]['input_dim'] = input_dim
            self.model = OrdinalClassifier(**config[args.reg_model])

        self.filter_all = filter_all
        self.use_onehot = use_onehot

        self.drop_list = config['base']['target_drop_list']
        self.train_task = args.train_task
        assert self.train_task in ['adr', 'revenue', 'label', 'pretrain']
        self.cancel_model = None
        if args.can_ckpt is not None:
            self.cancel_model = CancelModel('RFC', config, False, False)
            self.cancel_model.load(args.can_ckpt)

    def save(self, path):
        joblib.dump(self.__dict__, path)

    def train(self, X_tra, grader=None):
        if self.train_task == 'revenue':
            X_tra, Y_tra = get_revenue_pair(X_tra)
        elif self.train_task == 'adr':
            X_tra, _, Y_tra = get_adr_pair(X_tra)
        elif self.train_task == 'label':
            X_tra, Y_tra_adr, Y_tra_rev = get_rev_lst_pair(X_tra)
            Y_tra = (Y_tra_adr, Y_tra_rev)

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
                        return np.clip(self.model.predict(X_tst), a_max=370, a_min=-145) * (stays_in_weekend_nights + stays_in_week_nights) * cancel_pre
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
                # return [min(9.0, max(0.0, y.sum() // 10000)) for y in Y_pre]
                return [y.sum() / 10000 for y in Y_pre]

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
            def get_revenue(X_tst):
                if self.cancel_model is not None:
                    assert isinstance(X_tst, tuple) and len(X_tst) == 2
                    X_tst, X_tst_can = X_tst
                    place = []
                    counter = 0
                    for i in range(len(X_tst_can)):
                        place.append(
                            np.arange(len(X_tst_can[i])).astype(int) + counter)
                        counter += len(place[-1])

                    X_tst_can_cat = np.vstack([x for x in X_tst_can])
                    cancel_pre = 1 - self.cancel_model.predict(X_tst_can_cat)
                    cancel_pre = [cancel_pre[p] for p in place]
                SWDN = [x[:, -2] for x in X_tst]
                SWN = [x[:, -1] for x in X_tst]
                adr_pre = self.model.predict(X_tst)
                if self.cancel_model is not None:
                    return [adr_pre[i] * (SWDN[i] + SWN[i]) * cancel_pre[i] for i in range(len(adr_pre))]
                else:
                    return [adr_pre[i] * (SWDN[i] + SWN[i]) for i in range(len(adr_pre))]
            
            revenue = get_revenue(X_tst)
            if output == 'label':
                return [min(9.0, max(0.0, r.sum() // 10000)) for r in revenue]
            elif output == 'revenue':
                return revenue
         
    def load(self, path):
        if not self.IsStructLearning:
            params = joblib.load(path)
            for att in params:
                setattr(self, att, params[att])
