import argparse
import csv
import numpy as np
from os import makedirs, remove
from os.path import exists, join, basename, dirname
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.manifold import SpectralEmbedding
import torch
from torch.nn.utils.rnn import pad_sequence


class DataManager(object):
    def __init__(self, tra_path, tst_path):
        self.tra_df = self.read_csv(tra_path)
        self.tst_df = self.read_csv(tst_path)

        self.tra_partitions = self.tra_df.groupby(
            ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])
        self.tst_partitions = self.tst_df.groupby(
            ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])

        self.target_list = ['stays_in_weekend_nights',
                            'stays_in_week_nights', 'revenue', 'is_canceled', 'adr']
        self.filter_all = None
        self.all_cat_list = {}

        self.input_dim = None
        self.onehot_dim = None

    def get_revenue(self, x):
        """ build 'revenue' column
        """
        x['revenue'] = (x['stays_in_weekend_nights'] +
                        x['stays_in_week_nights']) * x['adr'] * (1 - x['is_canceled'])

    def read_csv(self, path):
        df = pd.read_csv(path).fillna(
            {'children': 0, 'agent': -1, 'company': -1, 'country': 'NAN'})
        df['agent'] = df['agent'].astype(int).astype(str).replace('-1', 'NAN')
        df['company'] = df['company'].astype(
            int).astype(str).replace('-1', 'NAN')
        df['arrival_date_month'] = df['arrival_date_month'].map(
            lambda s: month_converter(s) - 1)
        if 'adr' in df.columns:
            # df['adr'] = np.clip(df['adr'].values, a_max=370, a_min=-145)
            self.get_revenue(df)

        # Make the new column which contain 1 if guest received the same room which was reserved otherwise 0
        df['Room'] = 0
        df.loc[df['reserved_room_type'] ==
               df['assigned_room_type'], 'Room'] = 1

        # Make the new column which contain 1 if the guest has cancelled more booking in the past
        # than the number of booking he did not cancel, otherwise 0
        df['net_cancelled'] = 0
        df.loc[df['previous_cancellations'] >
               df['previous_bookings_not_canceled'], 'net_cancelled'] = 1

        return df

    def process_categories(self):
        # remove redundant categories
        columns = set(self.tra_df.columns) & set(self.tst_df.columns)
        self.all_cat_list = {}
        for c in columns:
            if not ('int' in str(self.tra_df[c].dtypes) or 'float' in str(self.tra_df[c].dtypes)):
                if self.filter_all:
                    self.all_cat_list[c] = (set(self.tra_df[c].values) & set(
                        self.tst_df[c].values)) - set(['NAN'])
                else:
                    self.all_cat_list[c] = set(
                        self.tra_df[c].values) - set(['NAN'])
                self.all_cat_list[c] = sorted(list(self.all_cat_list[c]))

                if len(set(self.tra_df[c].values) - set(self.all_cat_list[c])) != 0 or len(set(self.tst_df[c].values) - set(self.all_cat_list[c])) != 0:
                    self.all_cat_list[c] = ['<unknown>'] + self.all_cat_list[c]
            elif c == 'arrival_date_month':
                self.all_cat_list[c] = sorted(list(self.tra_df[c].values))

    def set_label_encoder(self):
        self.lb_enc = {cat: LabelEncoder().fit(
            self.all_cat_list[cat]) for cat in self.all_cat_list}

        self.lb2oh = {cat: LabelBinarizer().fit(self.lb_enc[cat].transform(
            self.all_cat_list[cat])) for cat in self.all_cat_list}

    def get_arr(self, df, use_onehot):

        # seperate different feature types
        text_list = []
        num_list = []
        for c in df.columns:
            if c not in self.target_list:
                if c in self.all_cat_list:
                    text_list.append(c)
                else:
                    num_list.append(c)

        # process category data
        category_data = []
        for cat in text_list:
            if use_onehot:
                # if cat != 'arrival_date_month':
                #     pseudo_cat = df[cat].value_counts().idxmax()
                #     pseudo_cat = pseudo_cat if pseudo_cat != 'NAN' else '<unknown>'
                #     df[cat] = df[cat].map(
                #         lambda s: pseudo_cat if (s not in self.all_cat_list[cat] or s == '<unknown>') else s)
                if cat != 'arrival_date_month':
                    df[cat] = df[cat].map(
                        lambda s: '<unknown>' if s not in self.all_cat_list[cat] else s)
            else:
                if cat != 'arrival_date_month':
                    pseudo_cat = df[cat].value_counts().idxmax()
                    pseudo_cat = pseudo_cat if pseudo_cat != 'NAN' else '<unknown>'
                    df[cat] = df[cat].map(
                        lambda s: pseudo_cat if (s not in self.all_cat_list[cat] or s == '<unknown>') else s)

            cat_feat = self.lb_enc[cat].transform(
                df[cat].values).reshape(-1, 1)

            if use_onehot:
                cat_feat = self.lb2oh[cat].transform(cat_feat)
                if '<unknown>' in self.all_cat_list[cat]:
                    label = self.lb_enc[cat].transform(['<unknown>'])[0]
                    cat_feat = cat_feat[:,
                                        1:] if label == 0 else cat_feat[:, :-1]
            category_data.append(cat_feat)
        category_data = np.hstack(category_data)
        self.onehot_dim = category_data.shape[-1]

        # process numeric data
        numeric_data = df[num_list].values

        target_list = self.target_list if 'adr' in df.columns else self.target_list[:-3]
        target_data = df[target_list].values

        data = np.concatenate(
            (category_data, numeric_data, target_data), axis=1)
        return data

    def get_feat(self, drop_list, filter_all, use_onehot, train=True):
        ''' Get the feature vectors grouped by dates
        return: (list)
            [[[2015, 7, 1], array(feature vectors)],
             [[2015, 7, 2], array(feature vectors)],
             ...
             [[2017, 3, 31], array(feature vectors)]]

        shape of feature vectors for each day: 
            (#requests, #features for each request)
        '''
        if train:
            df = self.tra_df.copy().drop(drop_list, axis=1)
        else:
            drop_list = list(set(drop_list) & set(self.tst_df.columns))
            df = self.tst_df.copy().drop(drop_list, axis=1)
      
        if self.filter_all != filter_all:
            self.filter_all = filter_all
            self.process_categories()
            self.set_label_encoder()
        data = self.get_arr(df, use_onehot)
        self.input_dim = data.shape[-1] - 3 if train else data.shape[-1]

        struc_data = []
        partitions = self.tra_partitions if train else self.tst_partitions
        for key in partitions.groups:
            struc_data.append([[key[0], key[1]+1, key[2]],
                               data[partitions.get_group(key).index]])
        #     if  key[0] == 2016 and key[1] in [3, 4, 5, 6, 7] and train:
        #         struc_data.append([[key[0], key[1]+1, key[2]],
        #                        data[partitions.get_group(key).index]])
        # if train:
        #     return struc_data
        return sorted(struc_data)

    def get_input_dim(self):
        return self.input_dim

    def get_onehot_dim(self):
        return self.onehot_dim


def get_revenue_pair(X):
    if isinstance(X, tuple):
        X, X_can = X
        X = np.vstack([x[1] for x in X])
        X_can = np.vstack([x[1] for x in X_can])
        return (X[:, :-3], X_can[:, :-3]), X[:, -3]
    else:
        X = np.vstack([x[1] for x in X])
        return X[:, :-3], X[:, -3]


def get_label_pair(X):
    if isinstance(X, tuple):
        X, X_can = X
        Y = [x[1][:, -3].sum() // 10000 for x in X]
        X = [[x[0], x[1][:, :-3]] for x in X]
        X_can = [[x[0], x[1][:, :-3]] for x in X_can]
        return (X, X_can), Y
    else:
        Y = [x[1][:, -3].sum() // 10000 for x in X]
        X = [[x[0], x[1][:, :-3]] for x in X]
        return X, Y


def get_rev_lst_pair(X):
    if isinstance(X, tuple):
        X, X_can = X
        rev = [x[1][:, -3] for x in X]
        adr = [x[1][:, -1] for x in X]
        X = [x[1][:, :-3] for x in X]
        X_can = [x[1][:, :-3] for x in X_can]
        return (X, X_can), adr, rev
    else:
        rev = [x[1][:, -3] for x in X]
        adr = [x[1][:, -1] for x in X]
        X = [x[1][:, :-3] for x in X]
        return X, adr, rev


def get_adr_pair(X):
    if isinstance(X, tuple):
        X, X_can = X
        X = np.vstack([x[1] for x in X])
        X_can = np.vstack([x[1] for x in X_can])
        return (X[:, :-3], X_can[:, :-3]), X[:, -2], X[:, -1]
    else:
        X = np.vstack([x[1] for x in X])
        # ['revenue', 'is_canceled', 'adr']
        return X[:, :-3], X[:, -2], X[:, -1]


def month_converter(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months.index(month[:3]) + 1


def write_test(X_tst, Y_pre, name):
    assert len(X_tst) == len(Y_pre)
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['arrival_date', 'label'])
        for i in range(len(X_tst)):
            x = X_tst[i]
            writer.writerow(['{:d}-{:02d}-{:02d}'.format(x[0][0],
                                                         x[0][1], x[0][2]), '{:.1f}'.format(Y_pre[i])])


class RegressorDataset(torch.utils.data.Dataset):
    def __init__(self, X_tra, Y_tra=None):
        super(RegressorDataset, self).__init__()
        self.X_tra = X_tra
        self.Y_tra = Y_tra

    def __getitem__(self, idx):
        if self.Y_tra is not None:
            return torch.from_numpy(np.array(self.X_tra[idx])).float(), \
                torch.from_numpy(np.array(self.Y_tra[idx])).float()
        else:
            return torch.from_numpy(np.array(self.X_tra[idx])).float()

    def __len__(self):
        return len(self.X_tra)


class OrdinalClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y=None):
        super(OrdinalClassifierDataset, self).__init__()
        self.X = X
        self.Y_adr = None
        self.Y_rev = None
            
        self.sample_interval = 1
        if Y is not None:
            Y_adr, Y_rev = Y
            self.Y_adr = Y_adr
            self.Y_rev = Y_rev
            self.min = min([len(x) for x in self.X])
            self.max = max([len(x) for x in self.X])
        
    def __getitem__(self, idx):
        if self.Y_adr is not None:
            X = [self.X[i] for i in range(
                max(0, idx-self.sample_interval), min(len(self.X), idx+self.sample_interval + 1))]
            Y_adr = [self.Y_adr[i] for i in range(
                max(0, idx-self.sample_interval), min(len(self.Y_adr), idx+self.sample_interval + 1))]
            Y_rev = [self.Y_rev[i] for i in range(
                max(0, idx-self.sample_interval), min(len(self.Y_rev), idx+self.sample_interval + 1))]

            X = np.vstack(X)
            Y_adr = np.hstack(Y_adr)
            Y_rev = np.hstack(Y_rev)

            random_list = np.random.permutation(len(Y_adr))
            num = np.random.randint(self.min, min(self.max, len(Y_adr)))

            X = X[random_list[:num]]
            Y_adr = Y_adr[random_list[:num]]
            Y_label = Y_rev[random_list[:num]].sum() // 10000

            return torch.from_numpy(np.array(X)).float(), \
                    torch.from_numpy(np.array(Y_adr)).float(), \
                    Y_label
        else:
            return torch.from_numpy(np.array(self.X[idx])).float()

    def collate_fn(self, samples):
        if self.Y_adr is not None:
            X = [s[0] for s in samples]
            Y_adr = [s[1] for s in samples]
            Y_label = [s[2] for s in samples]

            mask = self.length_to_mask(torch.LongTensor([len(x) for x in X]))
            X = pad_sequence(X, batch_first=True)
            Y_adr = pad_sequence(Y_adr, batch_first=True)
            Y_label = torch.Tensor(Y_label)
            return X, Y_adr, Y_label, mask
        else:
            X = [s for s in samples]
            mask = self.length_to_mask(torch.LongTensor([len(x) for x in X]))
            X = pad_sequence(X, batch_first=True)
            return X, mask

    def length_to_mask(self, length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask.unsqueeze(-1)

    def __len__(self):
        return len(self.X)
