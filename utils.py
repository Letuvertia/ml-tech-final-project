import argparse
import csv
import numpy as np
from os import makedirs, remove
from os.path import exists, join, basename, dirname
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
import torch


class DataManager(object):
    def __init__(self, path, drop_list=[]):
        self.df = self.read_csv(path)
        self.partitions = self.df.groupby(
            ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])
        self.drop_list = drop_list
        self.target_list = ['stays_in_weekend_nights',
                            'stays_in_week_nights', 'revenue', 'is_canceled', 'adr']
        #columns = [c for c in self.df.columns]
        #print(len(columns), columns)
        self.df = self.feature_filtering(self.df, self.drop_list)
        #columns = [c for c in self.df.columns]
        #print(len(columns), columns)
        self.set_onehot_encoder()
        

    def get_revenue(self, x):
        """ build 'revenue' column
        """
        x['revenue'] = (x['stays_in_weekend_nights'] +
                        x['stays_in_week_nights']) * x['adr'] * (1 - x['is_canceled'])


    def read_csv(self, path):
        df = pd.read_csv(path).fillna(
            {'children': 0, 'agent': -1, 'company': -1, 'country': 'NAN'})
        df['agent'] = df['agent'].astype(int).astype(str).replace('-1', 'NAN')
        df['company'] = df['company'].astype(int).astype(str).replace('-1', 'NAN')

        if 'adr' in df.columns:
            self.get_revenue(df)
        return df


    @staticmethod
    def feature_filtering(df, drop_list):
        return df.drop(set(drop_list) & set(df.columns), axis=1)


    def set_onehot_encoder(self):
        text_list = [c for c in self.df.columns if not ('int' in str(
            self.df[c].dtypes) or 'float' in str(self.df[c].dtypes))]
        
        self.oh_enc = OneHotEncoder()
        self.oh_enc.fit(self.df[text_list].values)


    def get_arr(self, df):
        text_list = []
        num_list = []
        for c in df.columns:
            if c not in self.target_list:
                if not ('int' in str(self.df[c].dtypes) or 'float' in str(self.df[c].dtypes)):
                    text_list.append(c)
                else:
                    num_list.append(c)
        print('*text column:', len(text_list), text_list, end='\n\n')
        print('*numeric column:', len(num_list), num_list, end='\n\n')
        
        category_data = self.oh_enc.transform(
            df[text_list].values).toarray()
        numeric_data = df[num_list].values
        
        # normalize numeric data
        # numeric_data = normalize(numeric_data, axis=0)
        
        if 'adr' in df.columns:
            target_data = df[self.target_list].values
        else:
            target_data = df[self.target_list[:-3]].values
        data = np.concatenate(
            (category_data, numeric_data, target_data), axis=1)
        return data


    def get_feat(self):
        ''' Get the feature vectors grouped by dates

        return: (list)
            [[[2015, 7, 1], array(feature vectors)],
             [[2015, 7, 2], array(feature vectors)],
             ...
             [[2017, 3, 31], array(feature vectors)]]
        
        shape of feature vectors for each day: 
            (#requests, #features for each request)
        '''
        data = self.get_arr(self.df)
        struc_data = []
        for key in self.partitions.groups:
            struc_data.append([[key[0], month_converter(
                key[1]), key[2]], data[self.partitions.get_group(key).index]])
        return sorted(struc_data)

    def load_test(self, tst_path):
        tst_df = self.read_csv(tst_path)
        tst_partitions = tst_df.groupby(
            ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])
        tst_df = self.feature_filtering(tst_df, self.drop_list)

        # filter unseen catgories
        for c in tst_df.columns:
            if not ('int' in str(self.df[c].dtypes) or 'float' in str(self.df[c].dtypes)):
                all_cat = set(self.df[c])
                tst_df[c][~tst_df[c].isin(all_cat)] = 'NAN'

        data = self.get_arr(tst_df)
        struc_data = []
        for key in tst_partitions.groups:
            struc_data.append([[key[0], month_converter(
                key[1]), key[2]], data[tst_partitions.get_group(key).index]])
        return sorted(struc_data)


def get_revenue_pair(X):
    X = np.vstack([x[1] for x in X])
    return X[:, :-3], X[:, -3]


def get_label_pair(X):
    Y = [x[1][:, -3].sum() // 10000 for x in X]
    X = [[x[0], x[1][:, :-3]] for x in X]
    return X, Y


def get_adr_pair(X):
    X = np.vstack([x[1] for x in X])
    return X[:, :-3], X[:, -2], X[:, -1] # ['revenue', 'is_canceled', 'adr']


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


def str_to_bool(v):
    """ This function turn all the True-intended string into True.

    Usage example:
        add_argument('--option1', type=str_to_bool, nargs='?', const=True, default=False, help='...')
    In cmd, all the following lines will be interpreted as option1=True:
        python train.py --option1        # this is set by (nargs='?', const=True)
        python train.py --option1 yes
        python train.py --option1 true
        python train.py --option1 t
        python train.py --option1 y
        python train.py --option1 1
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, file, is_best, is_final=False, is_training=True):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir != '' and not exists(model_dir):
        makedirs(model_dir)
    if is_training:
        torch.save(state, file)

    prefix = ''
    if is_final:
        prefix = 'best_'
    elif is_best:
        prefix = 'final_'
    torch.save(state, join(model_dir, prefix+model_fn))
