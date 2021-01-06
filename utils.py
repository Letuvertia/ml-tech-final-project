import argparse
import csv
import numpy as np
from os import makedirs, remove
from os.path import exists, join, basename, dirname

import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from random import shuffle, seed
import torch

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


class DataManager(object):
    def __init__(self, path, drop_list=[], 
                    poly_trans_list=[], poly_trans_degree=None, feature_copy_times=None):
        self.drop_list = drop_list
        self.poly_trans_list = poly_trans_list
        self.poly_trans_degree = poly_trans_degree
        self.feature_copy_times = feature_copy_times
        self.label_encoders = None
        self.scalers = None

        self.target_list = ['stays_in_weekend_nights', 'stays_in_week_nights',
                            'revenue', 'is_canceled', 'adr']
        
        self.df = self.read_csv(path)
        self.partitions = self.df.groupby(
            ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])
        self.df = self.feature_processing(self.df)
        #self.df = self.feature_engineering_preprocessed(path)
        #self.df = self.feature_engineering_original(self.df, path)
        #self.df = self.feature_engineering(self.df)
        # self.cancel_adr()
        
        #self.df = self.feature_filtering(self.df, self.drop_list)
        #self.set_onehot_encoder()

    
    def read_csv(self, path):
        df = pd.read_csv(path, index_col="ID")
        df = self.add_feature(df)
        return df
    
    def cancel_adr(self):
        anal_df = self.df.copy()
        mean_1 = anal_df.loc[anal_df['is_canceled']==1,'adr'].mean()
        std_1 = anal_df.loc[anal_df['is_canceled']==1,'adr'].std()
        mean_0 = anal_df.loc[anal_df['is_canceled']==0,'adr'].mean()
        std_0 = anal_df.loc[anal_df['is_canceled']==0,'adr'].std()
        print(f'[1] mean: {mean_1}; std: {std_1}')
        print(f'[0] mean: {mean_0}; std: {std_0}')
    

    def add_feature(self, df, revenue=True, net_cancelled=True):
        # revenue
        if revenue:
            df['revenue'] = (df['stays_in_weekend_nights'] +
                            df['stays_in_week_nights']) * df['adr'] * (1 - df['is_canceled'])
        # net_cancelled
        if net_cancelled:
            df['net_cancelled'] = 0
            df.loc[df['previous_cancellations'] > df['previous_bookings_not_canceled'], 'net_cancelled'] = 1
        
        return df
    

    def feature_processing(self, df):
        df = self.df.copy()

        df.arrival_date_month = df.arrival_date_month.map(MONTHS)
        df.children = df.children.fillna(0)
        nan_cols = self.get_columns_with_nan(df)

        for col in nan_cols:
            df[col] = df[col].fillna("Null").astype(str)
        
        df = self.feature_filtering(df, self.drop_list)

        df = self.label_encoder(df)
        df = self.use_scaler(df)
        if self.poly_trans_degree is not None:
            self.polynomial_transform(df, self.poly_trans_list, self.poly_trans_degree)
        if self.feature_copy_times is not None:
            self.feature_copy(df, self.poly_trans_list, self.feature_copy_times)

        return df

    
    @staticmethod
    def get_columns_with_nan(df):
        nan_values = df.isna()
        nan_columns = nan_values.any()
        columns_with_nan = df.columns[nan_columns].tolist()
        return columns_with_nan
    

    def label_encoder(self, df):
        print('*label encoded columns:', [cname for cname in df.columns if is_string_dtype(df[cname])], end='\n\n')
        encoders = {}
        for cname in df.columns:
            if is_string_dtype(df[cname]):
                encoders[cname] = LabelEncoder()
                df[cname] = encoders[cname].fit_transform(df[cname])
                df[cname] = df[cname].astype("category")
                #df[cname] = df[cname].astype("float")
        self.label_encoders = encoders
        return df


    def use_scaler(self, df):
        scalers = {}
        for cname in df.columns:
            if is_numeric_dtype(df[cname]) and cname not in ['revenue', 'is_canceled', 'adr']:
                scalers[cname] = MinMaxScaler(feature_range=(0, 1))
                df[cname] = scalers[cname].fit_transform(df[[cname]])
        self.scalers = scalers

        return df

    @staticmethod
    def train_test_split_by_date(
        all_data_list, partitions, val_ratio=0.25, random=False
    ):
        keys = [key for key in partitions.groups.keys()]
        train_amount = int(len(keys) * (1 - val_ratio))
        train_df = all_data_list[:train_amount]
        val_df = all_data_list[train_amount:]

        '''
        train_keys = keys[:train_amount]
        val_keys = keys[train_amount:]
        
        train_df = []
        for key in train_keys:
            for idx in partitions.get_group(key).index:
                print(idx)
                train_df.append(all_data_list[idx])

        val_df = []
        for key in val_keys:
            for idx in partitions.get_group(key).index:
                train_df.append(all_data_list[idx])
        '''

        print(f'*train size: {len(train_df)}; test size: {len(val_df)}', end='\n\n')
        if random:
            shuffle(train_df)
            shuffle(val_df)
        
        return train_df, val_df
    

    @staticmethod
    def feature_filtering(df, drop_list):
        return df.drop(set(drop_list) & set(df.columns), axis=1)


    def polynomial_transform(self, x, target_list, degree=1):
        if degree >= 2:
            for target_feature in target_list:
                for d in range(2, degree+1):
                    x['{}_poly_d{}'.format(target_feature, d)] = x[target_feature] ** d
    

    def feature_copy(self, x, target_list, copy_times=1):
        if copy_times >= 2:
            for target_feature in target_list:
                for d in range(2, copy_times+1):
                    x['{}_copy_d{}'.format(target_feature, d)] = x[target_feature]
    

    def feature_engineering_original(self, df, path):
        df = pd.read_csv(path).fillna(
            {'children': 0, 'agent': -1, 'company': -1, 'country': 'NAN'})
        df['agent'] = df['agent'].astype(int).astype(str).replace('-1', 'NAN')
        df['company'] = df['company'].astype(int).astype(str).replace('-1', 'NAN')

        if 'adr' in df.columns:
            self.get_revenue(df)
        return df
    

    def feature_engineering_preprocessed(self, path):
        df = pd.read_csv(path)
        if self.poly_trans_degree is not None:
            self.polynomial_transform(df, self.poly_trans_list, self.poly_trans_degree)

        if 'adr' in df.columns:
            df = self.add_feature(df)
        return df
    
    '''
    @staticmethod
    def feature_engineering(df):
        # reproduction of this feature pre-processing 
        # [https://medium.com/analytics-vidhya/exploratory-data-analysis-of-the-hotel-booking-demand-with-python-200925230106]

        # fill na and change data type
        df[['agent','company']] = df[['agent','company']].fillna(0.0)
        df['country'].fillna(df.country.mode().to_string(), inplace=True)
        df['children'].fillna(round(df.children.mean()), inplace=True)
        #df = df.drop(df[(df.adults+df.babies+df.children)==0].index)
        df[['children', 'company', 'agent']] = df[['children', 'company', 'agent']].astype('int64')
        
        # feature engineering - add new features
        df['Room'] = 0
        df.loc[df['reserved_room_type'] == df['assigned_room_type'], 'Room'] = 1
        
        df['net_cancelled'] = 0
        df.loc[df['previous_cancellations'] > df['previous_bookings_not_canceled'], 'net_cancelled'] = 1
        
        if 'adr' in df.columns:
            # build 'revenue' column
            df['revenue'] = (df['stays_in_weekend_nights'] +
                             df['stays_in_week_nights']) * df['adr'] * (1 - df['is_canceled'])

        return df

    @staticmethod
    def feature_label_encoder(df):
        label_enc = LabelEncoder()
        ## Select all categorcial features
        categorical_features = list(df.columns[df.dtypes == object])
        ## Apply Label Encoding on all categorical features
        return df[categorical_features].apply(lambda x: label_enc.fit_transform(x))
    
    def set_onehot_encoder(self):
        text_list = [c for c in self.df.columns if not ('int' in str(
            self.df[c].dtypes) or 'float' in str(self.df[c].dtypes))]
        
        self.oh_enc = OneHotEncoder()
        if text_list:
            self.oh_enc.fit(self.df[text_list].values)
    '''


    def get_arr(self, df):
        cat_list = []
        num_list = []
        for c in df.columns:
            if c not in self.target_list:
                if not is_numeric_dtype(self.df[c]):
                    cat_list.append(c)
                else:
                    num_list.append(c)
        print('*categorical column:\n', len(cat_list), cat_list, end='\n\n')
        print('*numeric column:\n', len(num_list), num_list, end='\n\n')
        
        #if text_list:
        #    category_data = self.oh_enc.transform(
        #        df[text_list].values).toarray()
        category_data = df[cat_list].values
        numeric_data = df[num_list].values
        
        # normalize numeric data
        # numeric_data = normalize(numeric_data, axis=0)
        
        '''
        if 'adr' in df.columns:
            target_data = df[self.target_list].values
        else:
            target_data = df[self.target_list[:-3]].values
        '''
        target_data = df[self.target_list].values

        if cat_list:
            data = np.concatenate(
                (category_data, numeric_data, target_data), axis=1)
        else:
            data = np.concatenate(
                (numeric_data, target_data), axis=1)
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
            #struc_data.append([[key[0], month_converter(
            #    key[1]), key[2]], data[self.partitions.get_group(key).index]])
            struc_data.append([[key[0], key[1], key[2]], 
                data[self.partitions.get_group(key).index]])
        return sorted(struc_data)


    def load_test(self, tst_path):
        tst_df = self.read_csv(tst_path)
        tst_df = self.feature_engineering(tst_df)
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
