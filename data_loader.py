import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from options import ArgumentParser

class DummyLoader(Dataset):
    """ Simply load all the data in the given csv (features and labels).
        You may do some preprocessing in __getitem__().

        *** Note that while this class is inherited from torch.utils.data.Dataset, you can still use it as a dataloader
            even if the model is not implemented with pytorch. Just do your preprocessing in __getitem__() and return numpy array.
            
            Usage example:
                class YourDataset(DummyLoader):
                    def __getitem__(self, idx):
                        # your pre-processing codes
                        return {'feature': feature, 'target':target}
                dataset = YourDataset(feature_file=....,
                                      label_file=....,
                                      )
                print(dataset[idx]) # this return {'feature': feature, 'target':target}

    
    args:
        :dataset_csv_path: (str)
            path to dataset directory
        :feature_file: (str)
            filename of feature csv
        :label_file: (str)
            filename of label_file
    """

    def __init__(self, dataset_csv_path, feature_file='', label_file='', transform=None):
        if feature_file != '':
            print('Loading feature data from:', os.path.join(dataset_csv_path, feature_file))
            self.train_feature = pd.read_csv(os.path.join(dataset_csv_path, feature_file))
        if label_file != '':
            print('Loading label data from:', os.path.join(dataset_csv_path, label_file))
            self.train_label = pd.read_csv(os.path.join(dataset_csv_path, label_file))
        print('dataset size: ', self.__len__())

        self.transform = transform
        self.col_names_list = list(self.train_feature)
    
    def __len__(self, return_train=True):
        if return_train: # return the number of all the requests (train)
            return len(self.train_feature)
        else: # return the number of days (test)
            return len(self.train_label)

    def __getitem__(self, idx):
        pass


class CancelDataset(DummyLoader):
    def __getitem__(self, idx):
        feature_col_names = ['arrival_date_week_number', 
                             'arrival_date_day_of_month',
                             'stays_in_weekend_nights',
                             'stays_in_week_nights',
                             'adults',
                             'children',
                             'babies',
                             'is_repeated_guest',
                             'previous_cancellations',
                             'previous_bookings_not_canceled']
        target_col_names = ['is_canceled']
        
        feature_col_idx = [list(self.col_names_list).index(col_n) for col_n in feature_col_names]
        target_col_idx = [list(self.col_names_list).index(col_n) for col_n in target_col_names]

        feature = torch.tensor(self.train_feature.iloc[idx, feature_col_idx].to_numpy().astype(np.float32))
        target = torch.tensor(self.train_feature.iloc[idx, target_col_idx].to_numpy().astype(np.float32))

        return {'feature': feature, 'target':target}


class YourDataset(DummyLoader):
    def __getitem__(self, idx):
        # TODO
        pass



# testing codes
if __name__ == '__main__':
    args,arg_groups = ArgumentParser().parse()
    print(args)
    print(arg_groups)

    dataset = CancelDataset(feature_file='train.csv',
                            label_file='train_label.csv',
                            **arg_groups['dataset'])
    print(dataset[1])

